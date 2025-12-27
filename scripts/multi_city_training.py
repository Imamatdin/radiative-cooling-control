"""
Multi-City Comprehensive RL Training for Datacenter Radiative Cooling.
FINAL VERSION - With Oracle Heuristic Baseline and Multi-Seed Training.

UPDATES FROM PREVIOUS VERSION:
1. Oracle Heuristic baseline with perfect forecast exploitation
2. Multi-seed training (3 seeds per algorithm) for confidence intervals
3. Reports mean ± std across seeds
4. All previous fixes retained (TD3/SAC target_q, reward shaping, etc.)

Uses REAL physics environment from src/environment/datacenter_env.py

Features:
- 3 cities: Phoenix (hot/dry), Houston (hot/humid), Seattle (mild/cloudy)
- 3 algorithms: DDPG, TD3, SAC (all forecast-aware)
- Oracle Heuristic baseline with perfect foresight
- 3 random seeds per configuration for statistical validity
- Proper train/test split: 10 months train, 2 months test (July + January)
- Ablation study: with vs without forecast (all cities)
- 1000 episodes, 48-hour periods

Estimated runtime: 6-8 hours on RTX 5090 (3 seeds × previous time)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time
import json
import os
import sys

# Ensure imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import REAL environment with full physics
from environment.datacenter_env import DatacenterCoolingEnv, load_weather


# =============================================================================
# CONFIGURATION
# =============================================================================

NUM_SEEDS = 3  # Number of random seeds for statistical validity
SEEDS = [42, 123, 456]  # Fixed seeds for reproducibility
NUM_EPISODES = 1000
WARMUP_STEPS = 3000
EPISODE_HOURS = 48
TEST_MONTHS = [1, 7]  # January and July held out

WEATHER_FILES = {
    'phoenix': 'data/weather/phoenix_az_tmy.csv',
    'houston': 'data/weather/houston_tx_tmy.csv',
    'seattle': 'data/weather/seattle_wa_tmy.csv',
}


# =============================================================================
# SEED MANAGEMENT
# =============================================================================

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# EXTENDED ENVIRONMENT WRAPPER
# =============================================================================

class TrainTestEnv:
    """Wrapper around DatacenterCoolingEnv with train/test split."""
    
    def __init__(self, weather_df, episode_hours=48, test_months=[1, 7]):
        self.base_env = DatacenterCoolingEnv(weather_df, episode_hours=episode_hours)
        self.weather_df = weather_df
        self.episode_hours = episode_hours
        self.test_months = test_months
        
        n_hours = len(weather_df)
        day_of_year = np.arange(n_hours) // 24
        self.month = (day_of_year % 365) // 30 + 1
        self.month = np.clip(self.month, 1, 12)
        
        max_start = n_hours - episode_hours - 24
        all_hours = np.arange(max_start)
        
        self.train_hours = all_hours[~np.isin(self.month[:max_start], test_months)]
        self.test_hours = all_hours[np.isin(self.month[:max_start], test_months)]
        
        if len(self.test_hours) == 0: self.test_hours = all_hours[:1000]
        if len(self.train_hours) == 0: self.train_hours = all_hours
    
    @property
    def state_dim(self):
        return self.base_env.state_dim
    
    @property
    def action_dim(self):
        return self.base_env.action_dim

    def reset(self, test=False):
        hours = self.test_hours if test else self.train_hours
        start_hour = np.random.choice(hours)
        return self.base_env.reset(start_hour=start_hour)
    
    def step(self, action):
        return self.base_env.step(action)
    
    def get_forecast(self, horizon=6):
        return self.base_env.get_forecast(horizon=horizon)


# =============================================================================
# ORACLE HEURISTIC CONTROLLER
# =============================================================================

class OracleHeuristic:
    """
    Oracle controller with perfect 6-hour forecast.
    Uses hand-tuned rules that exploit perfect foresight for:
    1. Radiative panel allocation based on current + future capacity
    2. Tank charging/discharging based on price arbitrage
    3. Pre-cooling before unfavorable conditions
    """
    
    def __init__(self):
        self.use_forecast = True  # For compatibility with evaluation
    
    def select_action(self, state, forecast=None, noise=0):
        """
        State indices (normalized 0-1):
        0: T_air (ambient temp)
        1: RH (relative humidity)  
        2: f_c (cloud fraction)
        3: h_day (hour of day)
        4: Q_rad_cap (radiative capacity)
        5: Q_load (cooling load)
        6: SOC (tank state of charge)
        7: p_elec (electricity price)
        8: is_night (binary)
        
        Forecast shape: (6, 4) - 6 hours × [temp, humidity, clouds, price]
        """
        # Current conditions
        rad_cap_norm = state[4]  # 0-1, multiply by ~500 for kW
        soc = state[6]  # Tank state of charge 0-1
        price_norm = state[7]  # Current price normalized
        is_night = state[8] > 0.5
        humidity = state[1]
        
        # Denormalize radiative capacity (assuming max ~500 kW)
        rad_cap_kW = rad_cap_norm * 500
        
        # === RADIATIVE FRACTION DECISION (α) ===
        # Use radiative cooling when capacity is good
        if rad_cap_kW > 300:
            alpha = 1.0  # Excellent conditions - use 100%
        elif rad_cap_kW > 200:
            alpha = 0.85
        elif rad_cap_kW > 100:
            alpha = 0.6
        elif rad_cap_kW > 50:
            alpha = 0.3
        else:
            alpha = 0.0  # Poor conditions - use tower only
        
        # Reduce radiative use in high humidity (less effective)
        if humidity > 0.7:
            alpha *= 0.7
        
        # === TANK CONTROL DECISION (β) ===
        # β < 0: charge (store cooling)
        # β > 0: discharge (use stored cooling)
        
        if forecast is not None and len(forecast) >= 6:
            # Perfect foresight: look at future prices and conditions
            future_prices = forecast[:, 3] if forecast.shape[1] > 3 else np.ones(6) * price_norm
            future_humidity = forecast[:, 1]
            future_temps = forecast[:, 0]
            
            avg_future_price = np.mean(future_prices)
            max_future_price = np.max(future_prices)
            avg_future_humidity = np.mean(future_humidity)
            
            # Price arbitrage with perfect foresight
            if price_norm < avg_future_price * 0.8:
                # Current price is cheap relative to future - CHARGE
                if soc < 0.95:
                    beta = -0.9  # Aggressive charging
                else:
                    beta = 0.0
            elif price_norm > avg_future_price * 1.2:
                # Current price is expensive relative to future - DISCHARGE
                if soc > 0.1:
                    beta = 0.8  # Discharge stored cooling
                else:
                    beta = 0.0
            else:
                # Moderate price difference
                if is_night and soc < 0.8 and rad_cap_kW > 150:
                    # Good nighttime radiative conditions - charge
                    beta = -0.5
                elif not is_night and soc > 0.3 and rad_cap_kW < 100:
                    # Daytime with poor radiative - discharge
                    beta = 0.4
                else:
                    beta = 0.0
            
            # Pre-cooling: if humidity spike coming, charge now
            if avg_future_humidity > humidity + 0.15 and soc < 0.7:
                beta = min(beta, -0.6)  # Ensure we're charging
                
        else:
            # No forecast - simple rule-based
            if is_night and soc < 0.8:
                beta = -0.5  # Charge at night
            elif not is_night and soc > 0.5:
                beta = 0.3  # Use during day
            else:
                beta = 0.0
        
        return np.array([alpha, beta])


# =============================================================================
# NEURAL NETWORKS
# =============================================================================

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        super().__init__()
        layers = []
        prev = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)
        self.net[-1].weight.data.uniform_(-3e-3, 3e-3)
        self.net[-1].bias.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, x):
        return torch.tanh(self.net(x))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        super().__init__()
        layers = []
        prev = state_dim + action_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, state, action):
        return self.net(torch.cat([state, action], -1))


class DoubleCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        super().__init__()
        self.q1 = Critic(state_dim, action_dim, hidden_dims)
        self.q2 = Critic(state_dim, action_dim, hidden_dims)
    
    def forward(self, state, action):
        return self.q1(state, action), self.q2(state, action)


# =============================================================================
# AGENTS
# =============================================================================

class BaseAgent:
    def __init__(self, state_dim, action_dim, use_forecast=True, forecast_dim=24,
                 hidden_dims=[256, 256], lr=3e-4, gamma=0.99, tau=0.005):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_forecast = use_forecast
        self.input_dim = state_dim + (forecast_dim if use_forecast else 0)
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.buffer = deque(maxlen=100000)
    
    def _process_state(self, state, forecast=None):
        if self.use_forecast and forecast is not None:
            return np.concatenate([state, forecast.flatten()])
        return state
    
    def store(self, state, forecast, action, reward, next_state, next_forecast, done):
        s = self._process_state(state, forecast)
        ns = self._process_state(next_state, next_forecast)
        self.buffer.append((s, action, reward, ns, done))
    
    def save(self, path):
        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}, path)
    
    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])


class DDPGAgent(BaseAgent):
    def __init__(self, state_dim=9, action_dim=2, use_forecast=True, **kwargs):
        super().__init__(state_dim, action_dim, use_forecast, **kwargs)
        
        self.actor = Actor(self.input_dim, action_dim).to(self.device)
        self.actor_target = Actor(self.input_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(self.input_dim, action_dim).to(self.device)
        self.critic_target = Critic(self.input_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=3e-4)
    
    def select_action(self, state, forecast=None, noise=0.1):
        s = self._process_state(state, forecast)
        s_t = torch.FloatTensor(s).unsqueeze(0).to(self.device)
        with torch.no_grad():
            a = self.actor(s_t).cpu().numpy()[0]
        
        if noise > 0:
            a = a + np.random.normal(0, noise, a.shape)
        
        # Scale: Rad [0,1], Tank [-1,1]
        a[0] = (a[0] + 1) / 2.0
        
        return np.clip(a, [0, -1], [1, 1])
    
    def train_step(self, batch_size=128):
        if len(self.buffer) < batch_size: return
        
        batch = random.sample(self.buffer, batch_size)
        s = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
        a = torch.FloatTensor(np.array([b[1] for b in batch])).to(self.device)
        r = torch.FloatTensor(np.array([b[2] for b in batch])).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(np.array([b[3] for b in batch])).to(self.device)
        d = torch.FloatTensor(np.array([b[4] for b in batch])).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            target_q = r + (1 - d) * self.gamma * self.critic_target(ns, self.actor_target(ns))
        
        critic_loss = nn.MSELoss()(self.critic(s, a), target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)


class TD3Agent(BaseAgent):
    def __init__(self, state_dim=9, action_dim=2, use_forecast=True, **kwargs):
        super().__init__(state_dim, action_dim, use_forecast, **kwargs)
        
        self.actor = Actor(self.input_dim, action_dim).to(self.device)
        self.actor_target = Actor(self.input_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = DoubleCritic(self.input_dim, action_dim).to(self.device)
        self.critic_target = DoubleCritic(self.input_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_delay = 2
        self.update_count = 0
    
    def select_action(self, state, forecast=None, noise=0.1):
        s = self._process_state(state, forecast)
        s_t = torch.FloatTensor(s).unsqueeze(0).to(self.device)
        with torch.no_grad():
            a = self.actor(s_t).cpu().numpy()[0]
        
        if noise > 0:
            a = a + np.random.normal(0, noise, a.shape)
        
        a[0] = (a[0] + 1) / 2.0
        
        return np.clip(a, [0, -1], [1, 1])
    
    def train_step(self, batch_size=128):
        if len(self.buffer) < batch_size: return
        self.update_count += 1
        
        batch = random.sample(self.buffer, batch_size)
        s = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
        a = torch.FloatTensor(np.array([b[1] for b in batch])).to(self.device)
        r = torch.FloatTensor(np.array([b[2] for b in batch])).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(np.array([b[3] for b in batch])).to(self.device)
        d = torch.FloatTensor(np.array([b[4] for b in batch])).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            na = (self.actor_target(ns) + noise).clamp(-1, 1)
            na[:, 0] = (na[:, 0] + 1) / 2.0
            
            tq1, tq2 = self.critic_target(ns, na)
            target_q = r + (1 - d) * self.gamma * torch.min(tq1, tq2)

        q1, q2 = self.critic(s, a)
        critic_loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        if self.update_count % self.policy_delay == 0:
            actor_loss = -self.critic.q1(s, self.actor(s)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)


class SACAgent(BaseAgent):
    def __init__(self, state_dim=9, action_dim=2, use_forecast=True, **kwargs):
        super().__init__(state_dim, action_dim, use_forecast, **kwargs)
        
        self.actor = Actor(self.input_dim, action_dim).to(self.device)
        self.critic = DoubleCritic(self.input_dim, action_dim).to(self.device)
        self.critic_target = DoubleCritic(self.input_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=3e-4)
    
    def select_action(self, state, forecast=None, noise=0.1):
        s = self._process_state(state, forecast)
        s_t = torch.FloatTensor(s).unsqueeze(0).to(self.device)
        with torch.no_grad():
            a = self.actor(s_t).cpu().numpy()[0]
        
        if noise > 0:
            a = a + np.random.normal(0, noise, a.shape)
        
        a[0] = (a[0] + 1) / 2.0
        
        return np.clip(a, [0, -1], [1, 1])
    
    def train_step(self, batch_size=128):
        if len(self.buffer) < batch_size: return
        
        batch = random.sample(self.buffer, batch_size)
        s = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
        a = torch.FloatTensor(np.array([b[1] for b in batch])).to(self.device)
        r = torch.FloatTensor(np.array([b[2] for b in batch])).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(np.array([b[3] for b in batch])).to(self.device)
        d = torch.FloatTensor(np.array([b[4] for b in batch])).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            na = self.actor(ns) + torch.randn_like(a) * 0.1
            na = na.clamp(-1, 1)
            na[:, 0] = (na[:, 0] + 1) / 2.0

            tq1, tq2 = self.critic_target(ns, na)
            target_q = r + (1 - d) * self.gamma * torch.min(tq1, tq2)

        q1, q2 = self.critic(s, a)
        critic_loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        actor_loss = -self.critic.q1(s, self.actor(s)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def evaluate(env, agent, num_episodes=20, test=True):
    """Evaluate agent and return detailed results."""
    results = []
    for _ in range(num_episodes):
        state = env.reset(test=test)
        forecast = env.get_forecast() if agent.use_forecast else None
        total_reward = 0
        tank_socs = []
        
        for _ in range(env.episode_hours):
            action = agent.select_action(state, forecast, noise=0)
            state, reward, done, info = env.step(action)
            
            forecast = env.get_forecast() if agent.use_forecast else None
            total_reward += reward
            tank_socs.append(info.get('tank_soc', 0.5))
            if done: break
        
        results.append({
            'reward': total_reward,
            'elec': info['electricity_savings_pct'],
            'water': info['water_savings_pct'],
            'tank_usage': np.mean(tank_socs)  # Mean SOC as tank usage metric
        })
    
    return {
        'reward_mean': np.mean([r['reward'] for r in results]),
        'reward_std': np.std([r['reward'] for r in results]),
        'elec_mean': np.mean([r['elec'] for r in results]),
        'elec_std': np.std([r['elec'] for r in results]),
        'water_mean': np.mean([r['water'] for r in results]),
        'water_std': np.std([r['water'] for r in results]),
        'tank_usage': np.mean([r['tank_usage'] for r in results]),
    }


def train_agent(env, agent, name, num_episodes=1000, warmup=3000, verbose=True):
    """Train a single agent and return training history."""
    if verbose:
        print(f"\n  Training {name}...")
    
    # Warmup with random actions
    state = env.reset(test=False)
    for _ in range(warmup):
        action = np.array([np.random.rand(), np.random.uniform(-1, 1)])
        forecast = env.get_forecast() if agent.use_forecast else None
        next_state, reward, done, _ = env.step(action)
        next_forecast = env.get_forecast() if agent.use_forecast else None
        agent.store(state, forecast, action, reward, next_state, next_forecast, done)
        state = next_state if not done else env.reset(test=False)
    
    if verbose:
        print(f"    Warmup done: {len(agent.buffer)} transitions")
    
    # Training loop
    rewards = []
    best_reward = -float('inf')
    
    for ep in range(1, num_episodes + 1):
        state = env.reset(test=False)
        forecast = env.get_forecast() if agent.use_forecast else None
        ep_reward = 0
        
        for _ in range(env.episode_hours):
            action = agent.select_action(state, forecast, noise=0.15)
            next_state, reward, done, info = env.step(action)
            next_forecast = env.get_forecast() if agent.use_forecast else None
            
            agent.store(state, forecast, action, reward, next_state, next_forecast, done)
            
            # Multiple gradient steps per env step
            for _ in range(2):
                agent.train_step()
            
            ep_reward += reward
            state, forecast = next_state, next_forecast
            if done: break
        
        rewards.append(ep_reward)
        
        if verbose and ep % 100 == 0:
            avg = np.mean(rewards[-100:])
            elec = info.get('electricity_savings_pct', 0.0)
            water = info.get('water_savings_pct', 0.0)
            print(f"    Ep {ep:4d}/{num_episodes} | Avg100: {avg:7.1f} | "
                  f"Elec: {elec:5.1f}% | Water: {water:5.1f}%")
            
            if avg > best_reward:
                best_reward = avg
    
    return rewards, best_reward


def aggregate_seed_results(seed_results):
    """Aggregate results across multiple seeds."""
    # Extract test results from each seed
    metrics = ['reward_mean', 'elec_mean', 'water_mean', 'tank_usage']
    
    aggregated = {}
    for metric in metrics:
        values = [sr['test'][metric] for sr in seed_results]
        base_metric = metric.replace('_mean', '')
        aggregated[f'{base_metric}_mean'] = float(np.mean(values))
        aggregated[f'{base_metric}_std'] = float(np.std(values))
    
    # Also store individual seed results
    aggregated['per_seed'] = seed_results
    
    return aggregated


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def main():
    print("=" * 80)
    print("MULTI-CITY RL TRAINING - FINAL VERSION")
    print("With Oracle Heuristic & Multi-Seed Training")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print(f"\nConfiguration:")
    print(f"  Seeds: {SEEDS}")
    print(f"  Episodes: {NUM_EPISODES}")
    print(f"  Episode length: {EPISODE_HOURS} hours")
    print(f"  Test months: {TEST_MONTHS} (held out)")
    
    cities = ['phoenix', 'houston', 'seattle']
    algorithms = ['DDPG', 'TD3', 'SAC']
    agent_classes = {'DDPG': DDPGAgent, 'TD3': TD3Agent, 'SAC': SACAgent}
    
    os.makedirs('results/models', exist_ok=True)
    
    all_results = {
        'training': {},
        'evaluation': {},
        'ablation': {},
        'oracle': {}
    }
    start_time = time.time()
    
    # ==========================================================================
    # PHASE 1: MULTI-SEED TRAINING FOR ALL ALGORITHMS
    # ==========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: MULTI-SEED TRAINING")
    print("=" * 80)
    
    for city in cities:
        print(f"\n{'='*60}")
        print(f"CITY: {city.upper()}")
        print(f"{'='*60}")
        
        weather_df = load_weather(WEATHER_FILES[city])
        print(f"  Loaded {len(weather_df)} hours from {WEATHER_FILES[city]}")
        
        for algo in algorithms:
            print(f"\n  Algorithm: {algo}")
            seed_results = []
            
            for seed_idx, seed in enumerate(SEEDS):
                print(f"    Seed {seed_idx+1}/{NUM_SEEDS} (seed={seed})")
                set_seed(seed)
                
                env = TrainTestEnv(weather_df, episode_hours=EPISODE_HOURS, test_months=TEST_MONTHS)
                
                agent = agent_classes[algo](
                    state_dim=9,
                    action_dim=2,
                    use_forecast=True,
                    forecast_dim=24
                )
                
                rewards, best = train_agent(
                    env, agent, f"{algo}_{city}_seed{seed}",
                    num_episodes=NUM_EPISODES, warmup=WARMUP_STEPS, verbose=False
                )
                
                # Save model for best seed (first one)
                if seed_idx == 0:
                    agent.save(f'results/models/{algo}_{city}.pt')
                
                test_results = evaluate(env, agent, num_episodes=20, test=True)
                
                seed_results.append({
                    'seed': seed,
                    'rewards': [float(r) for r in rewards[-100:]],  # Last 100 episodes
                    'best_reward': float(best),
                    'test': {k: float(v) for k, v in test_results.items()}
                })
                
                print(f"      Test: Elec={test_results['elec_mean']:.1f}%, "
                      f"Water={test_results['water_mean']:.1f}%")
            
            # Aggregate across seeds
            key = f"{algo}_{city}"
            all_results['evaluation'][key] = aggregate_seed_results(seed_results)
            
            agg = all_results['evaluation'][key]
            print(f"    {algo} AGGREGATED: Elec={agg['elec_mean']:.1f}% ± {agg['elec_std']:.1f}, "
                  f"Water={agg['water_mean']:.1f}% ± {agg['water_std']:.1f}")
    
    # ==========================================================================
    # PHASE 2: ORACLE HEURISTIC EVALUATION
    # ==========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: ORACLE HEURISTIC EVALUATION")
    print("=" * 80)
    
    oracle = OracleHeuristic()
    
    for city in cities:
        weather_df = load_weather(WEATHER_FILES[city])
        env = TrainTestEnv(weather_df, episode_hours=EPISODE_HOURS, test_months=TEST_MONTHS)
        
        test_results = evaluate(env, oracle, num_episodes=20, test=True)
        
        key = f"Oracle_{city}"
        all_results['evaluation'][key] = {
            'test': {k: float(v) for k, v in test_results.items()}
        }
        
        print(f"  {city}: Elec={test_results['elec_mean']:.1f}%, "
              f"Water={test_results['water_mean']:.1f}%")
    
    # ==========================================================================
    # PHASE 3: FIXED BASELINES
    # ==========================================================================
    print("\n" + "=" * 80)
    print("PHASE 3: FIXED BASELINES")
    print("=" * 80)
    
    baselines = {
        'Tower Only': np.array([0.0, 0.0]),
        'Fixed 50%': np.array([0.5, 0.0]),
        'Fixed 100%': np.array([1.0, 0.0])
    }
    
    for city in cities:
        weather_df = load_weather(WEATHER_FILES[city])
        env = TrainTestEnv(weather_df, episode_hours=EPISODE_HOURS, test_months=TEST_MONTHS)
        
        for bname, action_val in baselines.items():
            results = []
            for _ in range(20):
                state = env.reset(test=True)
                total_reward = 0
                for _ in range(env.episode_hours):
                    state, reward, done, info = env.step(action_val)
                    total_reward += reward
                    if done: break
                results.append({
                    'reward': total_reward,
                    'elec': info['electricity_savings_pct'],
                    'water': info['water_savings_pct']
                })
            
            key = f"{bname}_{city}"
            all_results['evaluation'][key] = {
                'test': {
                    'reward_mean': float(np.mean([r['reward'] for r in results])),
                    'reward_std': float(np.std([r['reward'] for r in results])),
                    'elec_mean': float(np.mean([r['elec'] for r in results])),
                    'elec_std': float(np.std([r['elec'] for r in results])),
                    'water_mean': float(np.mean([r['water'] for r in results])),
                    'water_std': float(np.std([r['water'] for r in results])),
                }
            }
        
        print(f"  {city}: Tower={all_results['evaluation'][f'Tower Only_{city}']['test']['water_mean']:.1f}%, "
              f"Fixed100={all_results['evaluation'][f'Fixed 100%_{city}']['test']['water_mean']:.1f}%")
    
    # ==========================================================================
    # PHASE 4: ABLATION STUDY (Single seed for speed)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("PHASE 4: ABLATION STUDY")
    print("=" * 80)
    
    set_seed(SEEDS[0])  # Use first seed for ablation
    
    for city in cities:
        print(f"\n  City: {city}")
        weather_df = load_weather(WEATHER_FILES[city])
        env = TrainTestEnv(weather_df, episode_hours=EPISODE_HOURS, test_months=TEST_MONTHS)
        
        for use_forecast in [True, False]:
            forecast_str = "forecast" if use_forecast else "no_forecast"
            name = f"DDPG_{forecast_str}_{city}"
            
            agent = DDPGAgent(
                state_dim=9,
                action_dim=2,
                use_forecast=use_forecast,
                forecast_dim=24 if use_forecast else 0
            )
            
            rewards, _ = train_agent(env, agent, name, num_episodes=500, warmup=2000, verbose=False)
            test_results = evaluate(env, agent, num_episodes=20, test=True)
            
            all_results['ablation'][name] = {
                'test': {k: float(v) for k, v in test_results.items()}
            }
            
            print(f"    {forecast_str}: Elec={test_results['elec_mean']:.1f}%, "
                  f"Water={test_results['water_mean']:.1f}%")
    
    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    total_time = time.time() - start_time
    all_results['meta'] = {
        'total_time_seconds': total_time,
        'total_time_hours': total_time / 3600,
        'cities': cities,
        'algorithms': algorithms,
        'seeds': SEEDS,
        'num_seeds': NUM_SEEDS,
        'episodes': NUM_EPISODES,
        'episode_hours': EPISODE_HOURS,
        'test_months': TEST_MONTHS,
        'device': str(device),
    }
    
    with open('results/multicity_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_time/3600:.2f} hours")
    
    print("\n" + "-" * 80)
    print("FINAL RESULTS (mean ± std across 3 seeds)")
    print("-" * 80)
    print(f"{'Config':<25} {'Elec Savings':>20} {'Water Savings':>20}")
    print("-" * 80)
    
    for city in cities:
        print(f"\n{city.upper()}")
        for algo in ['Tower Only', 'Fixed 50%', 'Fixed 100%', 'Oracle', 'DDPG', 'TD3', 'SAC']:
            key = f"{algo}_{city}"
            if key in all_results['evaluation']:
                r = all_results['evaluation'][key]
                if 'elec_std' in r:
                    # Multi-seed result
                    print(f"  {algo:<23} {r['elec_mean']:>8.1f}% ± {r['elec_std']:<6.1f} "
                          f"{r['water_mean']:>8.1f}% ± {r['water_std']:<6.1f}")
                else:
                    # Single result (baselines/oracle)
                    t = r['test']
                    print(f"  {algo:<23} {t['elec_mean']:>8.1f}%          "
                          f"{t['water_mean']:>8.1f}%")
    
    print("\n" + "-" * 80)
    print("ABLATION: Forecast Impact (DDPG)")
    print("-" * 80)
    for city in cities:
        with_f = all_results['ablation'].get(f'DDPG_forecast_{city}', {}).get('test', {})
        without_f = all_results['ablation'].get(f'DDPG_no_forecast_{city}', {}).get('test', {})
        if with_f and without_f:
            delta = with_f['water_mean'] - without_f['water_mean']
            print(f"  {city}: With={with_f['water_mean']:.1f}%, "
                  f"Without={without_f['water_mean']:.1f}%, Δ={delta:+.1f}pp")
    
    print("\n" + "=" * 80)
    print("Results saved to: results/multicity_results.json")
    print("Models saved to: results/models/")
    print("=" * 80)


if __name__ == "__main__":
    main()
