"""
Datacenter Cooling Environment (Production Master).
Target: 1 MW Google-Scale Pod with Thermal Battery.

Integrates:
- High-Efficiency Centrifugal Chiller (COP 6.5)
- Radiative Sky Cooling Panels (5000 m2)
- Stratified Water Thermal Storage (50 m3)
- Real Weather & Dynamic Pricing
- Full Reward Engineering (Time-of-Use Arbitrage)
"""

import numpy as np
import pandas as pd
import sys
import os

# Ensure we can import from src/physics
# This allows the script to find your physics modules no matter where it's run
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.physics.chiller import Chiller, CoolingTower, RadiativeHeatRejection
from src.physics.storage import ThermalStorageTank

class DatacenterCoolingEnv:
    def __init__(
        self,
        weather_df: pd.DataFrame,
        episode_hours: int = 48,
        electricity_price_default: float = 0.10,
        water_price: float = 0.005,
    ):
        self.weather_df = weather_df.reset_index(drop=True)
        self.episode_hours = int(episode_hours)
        self.water_price = float(water_price)
        
        # --- 1. REALISTIC "GOOGLE POD" SIZING (1 MW) ---
        # Chiller: 1000 kW, High Efficiency (COP 6.5)
        # (Claude's version was 400kW/5.5COP -> This is the "Pro" upgrade)
        self.chiller = Chiller(capacity_kW=1000, rated_COP=6.5)
        
        # Tower: 1500 kW (1.5x Safety Factor)
        self.tower = CoolingTower(capacity_kW=1500)
        
        # Panels: 5000 m2 (Approx 1 Football Field)
        # 5000m2 * 100 W/m2 = 500kW cooling capacity (50% of base load)
        self.panels = RadiativeHeatRejection(panel_area_m2=5000)
        
        # Battery: 50 m3 (50,000 Liters)
        # Stores ~300-400 kWh of cooling. 
        # Imported from src.physics.storage to ensure REAL thermodynamics.
        self.tank = ThermalStorageTank(
            volume_m3=50.0,
            T_initial_C=7.0,
            T_min_C=4.0,
            T_max_C=15.0
        )
        
        # 1 MW Base Load (Google Standard)
        self.base_cooling_load = 1000.0
        self.start_hour = 0
        self.current_step = 0
        
        # Pricing Structure (Time of Use) - EXACTLY AS CLAUDE DEFINED
        self.pricing = {
            'off_peak': 0.06,  # Night (Cheap)
            'mid_peak': 0.12,  # Day (Normal)
            'on_peak': 0.25    # 2PM - 7PM (Expensive)
        }
        
        # Tracking
        self.total_electricity_kWh = 0.0
        self.total_water_L = 0.0
        self.baseline_electricity_kWh = 0.0
        self.baseline_water_L = 0.0

    @property
    def state_dim(self):
        # 9 Features: Weather(3), Time(1), Cap(1), Load(1), SOC(1), Price(1), Night(1)
        return 9

    @property
    def action_dim(self):
        # 2 Actions: [Radiative_Fraction, Tank_Control]
        return 2

    @property
    def current_hour(self):
        return self.start_hour + self.current_step

    def get_electricity_price(self, hour: int) -> float:
        """Get electricity price based on time of day (TOU)."""
        hod = hour % 24
        # Off-peak: 10PM - 6AM
        if 22 <= hod or hod < 6: 
            return self.pricing['off_peak']
        # On-peak: 2PM - 7PM (Crucial for Battery Arbitrage)
        if 14 <= hod < 19: 
            return self.pricing['on_peak']
        # Mid-peak: Everything else
        return self.pricing['mid_peak']

    def reset(self, start_hour: int = None) -> np.ndarray:
        max_start = len(self.weather_df) - self.episode_hours - 24
        # Smart random start (avoid end of file)
        self.start_hour = start_hour if start_hour else np.random.randint(0, max(1, max_start))
        
        self.current_step = 0
        self.total_electricity_kWh = 0.0
        self.total_water_L = 0.0
        self.baseline_electricity_kWh = 0.0
        self.baseline_water_L = 0.0
        
        # Reset Tank to 50% SOC (Realistic random starting point)
        self.tank.reset(T_initial_C=np.random.uniform(6.0, 12.0))
        
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        w = self._get_weather(self.current_hour)
        hod = self.current_hour % 24
        price = self.get_electricity_price(self.current_hour)
        load = self._get_cooling_load(hod)
        
        rad_cap = self.panels.calculate_capacity(
            T_air_C=w['T_air_C'], rh=w['rh'], cloud_fraction=w['cloud'], T_fluid_in_C=35.0
        )
        
        # Normalized State Vector
        state = np.array([
            (w['T_air_C'] - 10) / 40,      # 0. Temp (Norm)
            w['rh'],                       # 1. RH (0-1)
            w['cloud'],                    # 2. Cloud (0-1)
            hod / 24.0,                    # 3. Time (0-1)
            rad_cap / 2000.0,              # 4. Rad Cap (Norm)
            load / 1500.0,                 # 5. Load (Norm)
            self.tank.state_of_charge,     # 6. SOC (0-1)
            (price - 0.06) / 0.19,         # 7. Price (Norm)
            1.0 if (hod < 6 or hod >= 20) else 0.0 # 8. Night Flag
        ], dtype=np.float32)
        return np.clip(state, 0.0, 1.0)

    def _get_weather(self, hour: int) -> dict:
        idx = min(hour, len(self.weather_df) - 1)
        row = self.weather_df.iloc[idx]
        return {
            'T_air_C': float(row['T_air_C']),
            'rh': float(np.clip(row['rh'], 0.01, 0.99)),
            'cloud': float(np.clip(row.get('cloud', 0.0), 0.0, 1.0))
        }

    def _get_cooling_load(self, hod: int) -> float:
        """
        Google Datacenter Load Profile (1 MW Base).
        Adds realistic daily variation + random noise.
        """
        base = 1000.0
        # Peak load during day (10AM - 6PM)
        profile = 0.85 + 0.15 * np.sin(((hod - 9) / 24) * 2 * np.pi)
        noise = np.random.uniform(-0.02, 0.02)
        return base * profile * (1 + noise)

    def step(self, action: np.ndarray) -> tuple:
        """
        Physics Step.
        Action[0]: Radiative Fraction (0-1)
        Action[1]: Tank Control (-1 to 1) -> (-) Charge, (+) Discharge
        """
        # Handle single float input or list
        if isinstance(action, (int, float)): action = [action, 0.0]
        
        rad_frac = np.clip(action[0], 0.0, 1.0)
        tank_ctrl = np.clip(action[1], -1.0, 1.0) 
        
        w = self._get_weather(self.current_hour)
        price = self.get_electricity_price(self.current_hour)
        load = self._get_cooling_load(self.current_hour % 24)
        
        # --- BASELINE CALCULATION (Standard Chiller + Tower) ---
        # This gives us the "Comparison" to prove savings
        base_tower = self.tower.operate(load * 1.3, w['T_air_C'], w['rh'])
        base_chiller = self.chiller.operate(load, base_tower['T_water_out_C'])
        base_cost = (base_chiller['W_electric_kW'] + base_tower['W_fan_kW']) * price + \
                    (base_tower['water_consumption_L'] * self.water_price)
        
        self.baseline_electricity_kWh += (base_chiller['W_electric_kW'] + base_tower['W_fan_kW'])
        self.baseline_water_L += base_tower['water_consumption_L']

        # --- HYBRID SYSTEM PHYSICS ---
        Q_discharge_kW = 0.0
        Q_charge_kW = 0.0
        
        if tank_ctrl > 0.05: # DISCHARGE (Help Chiller)
            # "tank_ctrl" determines % of max discharge rate (500 kW)
            req_Discharge = tank_ctrl * 500.0 
            res = self.tank.discharge(req_Discharge * 1000.0, dt_s=3600)
            Q_discharge_kW = res['energy_delivered_kWh'] # 1 hour = kWh
            
        elif tank_ctrl < -0.05: # CHARGE (Add to Load)
            # "tank_ctrl" determines % of max charge rate (500 kW)
            req_Charge = -tank_ctrl * 500.0
            res = self.tank.charge(req_Charge * 1000.0, dt_s=3600)
            Q_charge_kW = res['energy_stored_kWh']

        # Net Load the Chiller/Panels must handle
        # Net = DataCenter - Discharge(Help) + Charge(Burden)
        net_load = max(0, load - Q_discharge_kW + Q_charge_kW)
        
        # Total Heat to Reject (Load + Compressor Work Estimate ~30%)
        Q_total_reject = net_load * 1.3
        
        # Split heat between Radiative Panels and Tower
        Q_to_rad = Q_total_reject * rad_frac
        
        # 1. Run Panels
        rad_res = self.panels.operate(Q_to_rad, w['T_air_C'], w['rh'], w['cloud'], 35.0)
        
        # 2. Run Tower (Remainder)
        Q_remain = Q_total_reject - rad_res['Q_rejected_kW']
        if Q_remain > 0:
            tower_res = self.tower.operate(Q_remain, w['T_air_C'], w['rh'])
        else:
            tower_res = {'T_water_out_C': rad_res['T_fluid_out_C'], 'W_fan_kW': 0, 'water_consumption_L': 0}
            
        # 3. Mix Temperatures for Condenser
        # (Weighted average of Panel return and Tower return)
        if rad_res['Q_rejected_kW'] > 0 and Q_remain > 0:
            T_cond = (rad_res['Q_rejected_kW'] * rad_res['T_fluid_out_C'] + 
                      Q_remain * tower_res['T_water_out_C']) / Q_total_reject
        elif rad_res['Q_rejected_kW'] > 0:
            T_cond = rad_res['T_fluid_out_C']
        else:
            T_cond = tower_res['T_water_out_C']
            
        # 4. Run Chiller
        chiller_res = self.chiller.operate(net_load, T_cond)
        
        # --- COSTS & REWARDS ---
        elec_kW = chiller_res['W_electric_kW'] + tower_res['W_fan_kW'] + rad_res['W_pump_kW']
        water_L = tower_res['water_consumption_L']
        
        self.total_electricity_kWh += elec_kW
        self.total_water_L += water_L
        
        cost = elec_kW * price + water_L * self.water_price
        savings = base_cost - cost
        
        # --- REWARD ENGINEERING (PRO TUNING V3.4) ---
        # 1. Base Savings Reward (Increased multiplier for electricity focus)
        reward = savings * 15.0  

        # 2. PRO-FIX: Tank Stagnation Penalty
        # If SOC is stuck at 1.0 or 0.0 for too long, apply a small penalty
        # This forces the agent to realize "doing nothing" is not optimal.
        if self.tank.state_of_charge > 0.98 or self.tank.state_of_charge < 0.02:
            reward -= 0.5

        # 3. PRO-FIX: Arbitrage Multipliers
        # Bonus: Smart Charging (Charge when price is cheap)
        if tank_ctrl < -0.2 and price <= self.pricing['off_peak']:
            reward += 3.0 # Doubled incentive to charge at night
            
        # Bonus: Smart Discharging (Discharge when price is high)
        if tank_ctrl > 0.2 and price >= self.pricing['on_peak']:
            reward += 6.0 # Tripled incentive to discharge during peak
            
        # 4. Hard Grid Constraints
        # Penalty: Empty tank during Peak Pricing (Grid Stress)
        if price >= self.pricing['on_peak'] and self.tank.state_of_charge < 0.15:
            reward -= 10.0

        # Penalty: Dumb Charging (Charging during peak price)
        if tank_ctrl < -0.1 and price >= self.pricing['on_peak']:
            reward -= 10.0

        self.current_step += 1
        done = self.current_step >= self.episode_hours
        
        # --- Calculate Water Savings ---
        if self.baseline_water_L > 0:
            water_savings_pct = (self.baseline_water_L - self.total_water_L) / self.baseline_water_L * 100.0
        else:
            water_savings_pct = 0.0

        info = {
            'electricity_savings_pct': (self.baseline_electricity_kWh - self.total_electricity_kWh)/max(1, self.baseline_electricity_kWh)*100,
            'water_savings_pct': water_savings_pct,
            'tank_soc': self.tank.state_of_charge,
            'action_tank': tank_ctrl,
            'price': price
        }
        return self._get_state(), reward, done, info

    def get_forecast(self, horizon: int = 6) -> np.ndarray:
        forecast = []
        for h in range(horizon):
            w = self._get_weather(self.current_hour + h)
            price = self.get_electricity_price(self.current_hour + h)
            # Forecast Vector: [Temp, RH, Cloud, Normalized_Price]
            forecast.append([
                (w['T_air_C'] - 10) / 40, 
                w['rh'], 
                w['cloud'], 
                (price - 0.06) / 0.19
            ])
        return np.array(forecast, dtype=np.float32)

# Helper function to load and clean weather data
def load_weather(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower().str.strip()
    col_map = {'temp_air': 'T_air_C', 't_air': 'T_air_C', 'relative_humidity': 'rh', 'humidity': 'rh'}
    df = df.rename(columns=col_map)
    if 'rh' in df.columns and df['rh'].max() > 1.5: df['rh'] = df['rh'] / 100.0
    if 'cloud' not in df.columns: df['cloud'] = 0.0
    return df