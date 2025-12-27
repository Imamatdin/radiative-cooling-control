"""
Gymnasium-style environment for radiative cooling control.
IMPROVED VERSION with better reward shaping and realistic constraints.
"""

import numpy as np
from typing import Tuple, Dict, Optional
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from physics.radiation import selective_emitter_cooling_realistic
from physics.atmosphere import calculate_sky_conditions
from physics.panel import RadiativeCoolingPanel, convection_coefficient


class RadiativeCoolingEnv:
    """
    RL environment for adaptive radiative cooling control.
    
    IMPROVED: More realistic constraints and better reward shaping.
    """
    
    def __init__(
        self,
        panel_params: Optional[Dict] = None,
        T_inlet_C: float = 30.0,
        T_outlet_target_C: float = 28.0,
        T_outlet_max_C: float = 29.5,
        timestep: float = 60.0,
        episode_length: int = 480,
        normalize_state: bool = True,
    ):
        self.panel = RadiativeCoolingPanel(panel_params)
        self.T_inlet = T_inlet_C + 273.15
        self.T_outlet_target = T_outlet_target_C + 273.15
        self.T_outlet_max = T_outlet_max_C + 273.15
        self.timestep = timestep
        self.episode_length = episode_length
        self.normalize_state = normalize_state
        
        self.state_bounds = {
            'T_surface': (270.0, 320.0),
            'T_inlet': (290.0, 310.0),
            'T_sky': (240.0, 310.0),
            'T_air': (270.0, 320.0),
            'q_rad': (0.0, 150.0),
        }
        
        self.state_dim = 5
        self.action_dim = 1
        self.current_step = 0
        self.weather_profile = None
        self.done = False
        
        self.episode_metrics = {
            'total_cooling': 0.0,
            'total_pump_energy': 0.0,
            'violations': 0,
            'operating_steps': 0,
        }
        
    def _generate_weather_profile(self, scenario: str = 'variable') -> np.ndarray:
        n = self.episode_length
        
        if scenario == 'clear':
            T_air = 22.0 + 3.0 * np.sin(np.linspace(0, np.pi, n))
            rh = 0.25 + 0.05 * np.random.randn(n)
            clouds = 0.05 * np.abs(np.random.randn(n))
            wind = 2.0 + 0.5 * np.random.randn(n)
        elif scenario == 'cloudy':
            T_air = 20.0 + 2.0 * np.sin(np.linspace(0, np.pi, n))
            rh = 0.50 + 0.05 * np.random.randn(n)
            clouds = np.zeros(n)
            clouds[n//4:3*n//4] = 0.6 * np.sin(np.linspace(0, np.pi, n//2))
            wind = 1.5 + 0.3 * np.random.randn(n)
        elif scenario == 'humid':
            T_air = 26.0 + 2.0 * np.sin(np.linspace(0, np.pi, n))
            rh = 0.75 + 0.05 * np.random.randn(n)
            clouds = 0.2 + 0.1 * np.abs(np.random.randn(n))
            wind = 1.0 + 0.3 * np.random.randn(n)
        elif scenario == 'variable':
            T_air = 18.0 + 6.0 * np.random.rand() + 3.0 * np.sin(np.linspace(0, np.pi, n))
            rh_base = 0.3 + 0.4 * np.random.rand()
            rh = rh_base + 0.1 * np.cumsum(np.random.randn(n)) / np.sqrt(n)
            cloud_base = 0.2 * np.random.rand()
            clouds = cloud_base + 0.15 * np.abs(np.sin(np.linspace(0, 2*np.pi, n)))
            wind = 1.0 + 2.0 * np.random.rand() + 0.5 * np.random.randn(n)
        elif scenario == 'marginal':
            T_air = 25.0 + 2.0 * np.sin(np.linspace(0, np.pi, n))
            rh = 0.65 + 0.1 * np.random.randn(n)
            clouds = 0.1 * np.ones(n)
            clouds[n//3:2*n//3] = 0.5 + 0.2 * np.sin(np.linspace(0, np.pi, n//3))
            wind = 1.5 + 0.3 * np.random.randn(n)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
        
        rh = np.clip(rh, 0.1, 0.95)
        clouds = np.clip(clouds, 0.0, 1.0)
        wind = np.clip(wind, 0.1, 10.0)
        
        return np.stack([T_air, rh, clouds, wind], axis=1)
    
    def _get_weather(self, step: int) -> Dict:
        T_air_C, rh, clouds, wind = self.weather_profile[step]
        sky = calculate_sky_conditions(T_air_C, rh, clouds)
        return {
            'T_air_C': T_air_C,
            'T_air_K': T_air_C + 273.15,
            'relative_humidity': rh,
            'cloud_fraction': clouds,
            'wind_speed': wind,
            'T_sky_K': sky['T_sky_K'],
            'T_sky_C': sky['T_sky_C'],
        }
    
    def _calculate_radiative_cooling(self, weather: Dict) -> float:
        result = selective_emitter_cooling_realistic(
            T_surface_K=self.panel.T_surface,
            T_sky_K=weather['T_sky_K'],
            T_air_K=weather['T_air_K'],
            relative_humidity=weather['relative_humidity'],
            cloud_fraction=weather['cloud_fraction'],
        )
        return max(0.0, result['q_total'])
    
    def _normalize_state(self, state_raw: np.ndarray) -> np.ndarray:
        bounds = [
            self.state_bounds['T_surface'],
            self.state_bounds['T_inlet'],
            self.state_bounds['T_sky'],
            self.state_bounds['T_air'],
            self.state_bounds['q_rad'],
        ]
        state_norm = np.zeros_like(state_raw)
        for i, (low, high) in enumerate(bounds):
            state_norm[i] = np.clip((state_raw[i] - low) / (high - low), 0.0, 1.0)
        return state_norm
    
    def _get_state(self, weather: Dict, q_rad: float) -> np.ndarray:
        state_raw = np.array([
            self.panel.T_surface,
            self.T_inlet,
            weather['T_sky_K'],
            weather['T_air_K'],
            q_rad,
        ])
        if self.normalize_state:
            return self._normalize_state(state_raw)
        return state_raw
    
    def _action_to_flow_rate(self, action: float) -> float:
        action = np.clip(action, 0.0, 1.0)
        flow_range = self.panel.flow_rate_max - self.panel.flow_rate_min
        return self.panel.flow_rate_min + action * flow_range
    
    def _calculate_reward(self, Q_delivered: float, flow_rate: float, T_outlet: float, q_rad: float) -> Tuple[float, Dict]:
        max_possible_cooling = self.panel.area * q_rad
        if max_possible_cooling > 0:
            cooling_efficiency = Q_delivered / max_possible_cooling
        else:
            cooling_efficiency = 0.0
        cooling_efficiency = np.clip(cooling_efficiency, 0.0, 1.0)
        r_cooling = 2.0 * cooling_efficiency
        
        if T_outlet <= self.T_outlet_target:
            r_temp = 0.5
        elif T_outlet <= self.T_outlet_max:
            r_temp = 0.5 - 2.0 * (T_outlet - self.T_outlet_target) / (self.T_outlet_max - self.T_outlet_target)
        else:
            violation = T_outlet - self.T_outlet_max
            r_temp = -1.5 - 3.0 * violation
        
        pump_power_normalized = (flow_rate / self.panel.flow_rate_max) ** 2
        r_pump = -0.1 * pump_power_normalized
        
        if Q_delivered > 100:
            r_operating = 0.3
        elif Q_delivered > 0:
            r_operating = 0.1
        else:
            r_operating = -0.2
        
        reward = r_cooling + r_temp + r_pump + r_operating
        
        breakdown = {
            'r_cooling': r_cooling,
            'r_temp': r_temp,
            'r_pump': r_pump,
            'r_operating': r_operating,
            'Q_delivered': Q_delivered,
            'cooling_efficiency': cooling_efficiency,
            'T_outlet': T_outlet,
            'violation': max(0.0, T_outlet - self.T_outlet_max),
        }
        return reward, breakdown
    
    def reset(self, scenario: str = 'variable', seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        
        self.weather_profile = self._generate_weather_profile(scenario)
        self.panel.reset(T_initial=self.T_inlet - 2)
        self.current_step = 0
        self.done = False
        self.episode_metrics = {
            'total_cooling': 0.0,
            'total_pump_energy': 0.0,
            'violations': 0,
            'operating_steps': 0,
        }
        
        weather = self._get_weather(0)
        q_rad = self._calculate_radiative_cooling(weather)
        return self._get_state(weather, q_rad)
    
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.done:
            raise RuntimeError("Episode done. Call reset().")
        
        flow_rate = self._action_to_flow_rate(action)
        weather = self._get_weather(self.current_step)
        q_rad = self._calculate_radiative_cooling(weather)
        h_conv = convection_coefficient(weather['wind_speed'])
        
        panel_result = self.panel.step(
            T_fluid_in=self.T_inlet,
            q_net=q_rad,
            flow_rate=flow_rate,
            T_air=weather['T_air_K'],
            h_conv=h_conv,
            dt=self.timestep,
        )
        
        reward, reward_info = self._calculate_reward(
            Q_delivered=panel_result['Q_delivered'],
            flow_rate=flow_rate,
            T_outlet=panel_result['T_fluid_out'],
            q_rad=q_rad,
        )
        
        self.episode_metrics['total_cooling'] += panel_result['Q_delivered'] * self.timestep / 3600
        pump_power = 50.0 * (flow_rate / self.panel.flow_rate_max) ** 3
        self.episode_metrics['total_pump_energy'] += pump_power * self.timestep / 3600
        
        if reward_info['violation'] > 0:
            self.episode_metrics['violations'] += 1
        if panel_result['Q_delivered'] > 0:
            self.episode_metrics['operating_steps'] += 1
        
        self.current_step += 1
        self.done = self.current_step >= self.episode_length
        
        if not self.done:
            next_weather = self._get_weather(self.current_step)
            next_q_rad = self._calculate_radiative_cooling(next_weather)
            next_state = self._get_state(next_weather, next_q_rad)
        else:
            next_state = np.zeros(self.state_dim)
        
        info = {**weather, **panel_result, **reward_info, 'q_rad': q_rad, 'step': self.current_step}
        return next_state, reward, self.done, info
    
    def get_episode_summary(self) -> Dict:
        m = self.episode_metrics
        return {
            'total_cooling_Wh': m['total_cooling'],
            'total_pump_Wh': m['total_pump_energy'],
            'net_cooling_Wh': m['total_cooling'] - m['total_pump_energy'],
            'violations': m['violations'],
            'violation_rate': m['violations'] / max(1, self.episode_length),
            'operating_rate': m['operating_steps'] / max(1, self.episode_length),
        }


class FixedFlowController:
    def __init__(self, flow_rate_normalized: float = 0.5):
        self.flow_rate = flow_rate_normalized
    def act(self, state: np.ndarray) -> float:
        return self.flow_rate


class ProportionalController:
    def __init__(self, k_p: float = 1.0, flow_min: float = 0.1):
        self.k_p = k_p
        self.flow_min = flow_min
    def act(self, state: np.ndarray) -> float:
        q_rad_norm = state[4]
        flow = self.flow_min + self.k_p * q_rad_norm * (1.0 - self.flow_min)
        return np.clip(flow, 0.0, 1.0)


class AdaptiveController:
    def __init__(self, target_delta_T: float = 2.0):
        self.target_delta_T = target_delta_T
        self.flow_rate = 0.5
    def act(self, state: np.ndarray) -> float:
        q_rad_norm = state[4]
        if q_rad_norm > 0.5:
            self.flow_rate = min(1.0, self.flow_rate + 0.02)
        elif q_rad_norm > 0.3:
            pass
        elif q_rad_norm > 0.1:
            self.flow_rate = max(0.1, self.flow_rate - 0.02)
        else:
            self.flow_rate = 0.1
        return self.flow_rate


if __name__ == "__main__":
    print("=" * 70)
    print("IMPROVED ENVIRONMENT - VALIDATION")
    print("=" * 70)
    
    env = RadiativeCoolingEnv(episode_length=240)
    controllers = {
        'Fixed 30%': FixedFlowController(0.3),
        'Fixed 50%': FixedFlowController(0.5),
        'Fixed 80%': FixedFlowController(0.8),
        'Proportional': ProportionalController(),
        'Adaptive': AdaptiveController(),
    }
    scenarios = ['clear', 'cloudy', 'humid', 'marginal']
    
    print(f"\n{'Scenario':<12} {'Controller':<15} {'Cooling (Wh)':<14} {'Violations':<12} {'Reward'}")
    print("-" * 70)
    
    for scenario in scenarios:
        for name, ctrl in controllers.items():
            state = env.reset(scenario=scenario, seed=42)
            total_reward = 0
            done = False
            while not done:
                action = ctrl.act(state)
                state, reward, done, info = env.step(action)
                total_reward += reward
            summary = env.get_episode_summary()
            print(f"{scenario:<12} {name:<15} {summary['total_cooling_Wh']:<14.1f} "
                  f"{summary['violations']:<12} {total_reward:.1f}")
        print()
    print("=" * 70)
