"""
Datacenter cooling environment: 1 MW IT load, chiller + radiative panel field + cooling tower,
with a stratified chilled-water storage tank.

Corrected version (Oct 2026). Changes from the version used for the original paper draft:

1. Sky radiation comes from the downwelling longwave in the weather files (PVGIS 'IR(h)', ERA5).
   The old loader had no cloud column and silently set cloud = 0, so every hour was clear-sky.
2. Weather rows are UTC; they are now shifted to local standard time. Time-of-use prices,
   the night flag, the load profile and hour-of-day were previously applied to UTC hours.
3. Plant physics is solved as one energy-consistent condenser loop (src/physics/condenser_loop.py).
   The old panel model could cool the fluid below ambient through a surface held at 33 C.
4. Daytime solar absorption on the panels and wind-dependent convection are included.
5. Tank minimum temperature is the chilled-water supply temperature (7 C): a chiller with a
   7 C evaporator cannot charge the tank to 4 C.
6. Chiller capacity is 1.2 MW so peak load plus tank charging is physically servable; charging
   is limited to chiller headroom and any unmet cooling is reported in info.
7. Reward = cost savings vs tower-only. The hand-written arbitrage bonuses are off by default
   (reward_shaping=True restores them for ablation) because they encode the storage strategy
   the agent is claimed to discover.

Interface is unchanged: reset(start_hour) -> state, step(action) -> (state, reward, done, info),
get_forecast(horizon), state_dim = 9, action_dim = 2.
"""

import os
import numpy as np
import pandas as pd

from src.physics.condenser_loop import PlantConfig, plant_hour, panel_capacity_kW, SIGMA
from src.physics.storage import ThermalStorageTank

# Local standard-time offsets from UTC (hours). Tariffs follow local prevailing time; DST is
# ignored here (stated as a limitation), which shifts summer TOU windows by one hour.
CITY_UTC_OFFSET = {"phoenix": -7, "houston": -6, "seattle": -8}

PRICES = {"off_peak": 0.06, "mid_peak": 0.12, "on_peak": 0.25}   # $/kWh, illustrative TOU tariff
WATER_PRICE = 0.005                                              # $/L


def tou_price(hod):
    hod = np.asarray(hod) % 24
    return np.where((hod >= 22) | (hod < 6), PRICES["off_peak"],
                    np.where((hod >= 14) & (hod < 19), PRICES["on_peak"], PRICES["mid_peak"]))


def it_cooling_load(hod, base_kW=1000.0):
    """Deterministic daily load profile [kW] (peak mid-afternoon), local hour of day."""
    return base_kW * (0.85 + 0.15 * np.sin(((np.asarray(hod, float) - 9) / 24) * 2 * np.pi))


def load_weather(filepath: str, utc_offset_h: int = None) -> pd.DataFrame:
    """
    Load a PVGIS TMY csv (pvlib get_pvgis_tmy output) and return hourly local-standard-time rows with
    T_air_C, rh (0-1), ghi, L_down (downwelling longwave W/m2), wind (m/s).
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower().str.strip()
    required = {"temp_air", "relative_humidity", "ghi", "ir(h)", "wind_speed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{filepath}: missing columns {sorted(missing)}; expected PVGIS TMY format")
    if utc_offset_h is None:
        name = os.path.basename(filepath).lower()
        matches = [off for city, off in CITY_UTC_OFFSET.items() if city in name]
        if not matches:
            raise ValueError(f"cannot infer UTC offset from '{name}'; pass utc_offset_h")
        utc_offset_h = matches[0]
    out = pd.DataFrame({
        "T_air_C": df["temp_air"].astype(float).values,
        "rh": np.clip(df["relative_humidity"].astype(float).values / 100.0, 0.01, 0.99),
        "ghi": np.clip(df["ghi"].astype(float).values, 0, None),
        "L_down": df["ir(h)"].astype(float).values,
        "wind": np.clip(df["wind_speed"].astype(float).values, 0, None),
    })
    # local[i] = utc[i - offset]  (offset negative for the Americas)
    out = pd.DataFrame(np.roll(out.values, utc_offset_h, axis=0), columns=out.columns)
    out.attrs["utc_offset_h"] = utc_offset_h
    return out


class DatacenterCoolingEnv:
    def __init__(self, weather_df: pd.DataFrame, episode_hours: int = 48,
                 plant_cfg: PlantConfig = None, reward_shaping: bool = False,
                 load_noise: bool = True, water_price: float = WATER_PRICE,
                 tank_volume_m3: float = 50.0, tank_power_kW: float = 500.0, reward_scale: float = 15.0):
        self.weather_df = weather_df.reset_index(drop=True)
        self.n_hours = len(self.weather_df)
        self.episode_hours = int(episode_hours)
        self.cfg = plant_cfg or PlantConfig()
        self.reward_shaping = reward_shaping
        self.load_noise = load_noise
        self.water_price = float(water_price)
        self.tank_power_kW = float(tank_power_kW)
        self.reward_scale = float(reward_scale)
        self.tank = ThermalStorageTank(volume_m3=tank_volume_m3, T_initial_C=11.0, T_min_C=7.0, T_max_C=15.0)
        self.pricing = PRICES

        w = self.weather_df
        self._wx = {k: w[k].values.astype(float) for k in ("T_air_C", "rh", "ghi", "L_down", "wind")}
        # Precompute per-hour features (vectorized)
        self._rad_cap = panel_capacity_kW(self._wx, self.cfg)
        self._sky_deficit = SIGMA * (self._wx["T_air_C"] + 273.15) ** 4 - self._wx["L_down"]

        self.start_hour = 0
        self.current_step = 0
        self._reset_tracking()

    # ---- helpers used by baselines -------------------------------------------------
    def wx_at(self, hour):
        i = np.asarray(hour) % self.n_hours
        return {k: v[i] for k, v in self._wx.items()}

    def get_electricity_price(self, hour):
        return float(tou_price(hour % 24))

    def expected_load(self, hour):
        return it_cooling_load(np.asarray(hour) % 24)

    @property
    def tank_stored_kWh(self):
        return self.tank.available_cooling_kWh

    @property
    def tank_capacity_kWh(self):
        return self.tank.capacity_kWh

    def hour_cost(self, out, price):
        return out["W_total_kW"] * price + out["water_L"] * self.water_price

    # ---- gym-style API --------------------------------------------------------------
    @property
    def state_dim(self):
        return 9

    @property
    def action_dim(self):
        return 2

    @property
    def current_hour(self):
        return self.start_hour + self.current_step

    def _reset_tracking(self):
        self.total_electricity_kWh = 0.0
        self.total_water_L = 0.0
        self.total_cost = 0.0
        self.baseline_electricity_kWh = 0.0
        self.baseline_water_L = 0.0
        self.baseline_cost = 0.0
        self.unmet_cooling_kWh = 0.0

    def reset(self, start_hour: int = None, soc0: float = None) -> np.ndarray:
        max_start = self.n_hours - self.episode_hours - 24
        self.start_hour = int(start_hour) if start_hour is not None else int(np.random.randint(0, max(1, max_start)))
        self.current_step = 0
        self._reset_tracking()
        if soc0 is None:
            T0 = np.random.uniform(8.0, 14.0)
        else:
            T0 = 15.0 - float(np.clip(soc0, 0, 1)) * (15.0 - 7.0)
        self.tank.reset(T_initial_C=T0)
        self.tank_start_kWh = self.tank_stored_kWh
        return self._get_state()

    def _load(self, hour):
        load = float(it_cooling_load(hour % 24))
        if self.load_noise:
            load *= 1 + np.random.uniform(-0.02, 0.02)
        return load

    def _get_state(self) -> np.ndarray:
        h = self.current_hour % self.n_hours
        hod = self.current_hour % 24
        price = self.get_electricity_price(self.current_hour)
        state = np.array([
            (self._wx["T_air_C"][h] - 10) / 40,
            self._wx["rh"][h],
            self._sky_deficit[h] / 150.0,          # sky radiative deficit (replaces the always-zero cloud slot)
            hod / 24.0,
            self._rad_cap[h] / 2000.0,
            float(it_cooling_load(hod)) / 1500.0,
            self.tank.state_of_charge,
            (price - 0.06) / 0.19,
            1.0 if (hod < 6 or hod >= 20) else 0.0,
        ], dtype=np.float32)
        return np.clip(state, 0.0, 1.0)

    def step(self, action):
        if isinstance(action, (int, float)):
            action = [action, 0.0]
        alpha = float(np.clip(action[0], 0.0, 1.0))
        beta = float(np.clip(action[1], -1.0, 1.0))
        hour = self.current_hour
        wx = self.wx_at(hour)
        price = self.get_electricity_price(hour)
        load = self._load(hour)
        cap = self.cfg.chiller_capacity_kW

        Q_dis = Q_ch = 0.0
        if beta > 0.05:
            Q_dis = self.tank.discharge(beta * self.tank_power_kW * 1000.0, dt_s=3600)["energy_delivered_kWh"]
            Q_dis = min(Q_dis, load)
        elif beta < -0.05:
            headroom = max(0.0, cap - load)
            req = min(-beta * self.tank_power_kW, headroom)
            if req > 0:
                Q_ch = self.tank.charge(req * 1000.0, dt_s=3600)["energy_stored_kWh"]
        Q_evap = load - Q_dis + Q_ch
        unmet = max(0.0, Q_evap - cap)
        Q_evap = min(Q_evap, cap)
        self.unmet_cooling_kWh += unmet

        out = {k: float(v) for k, v in plant_hour(Q_evap, alpha, wx, self.cfg).items()}
        base = {k: float(v) for k, v in plant_hour(min(load, cap), 0.0, wx, self.cfg).items()}
        cost = self.hour_cost(out, price) + unmet * 10.0          # unmet cooling is heavily penalised
        base_cost = self.hour_cost(base, price)

        self.total_electricity_kWh += out["W_total_kW"]
        self.total_water_L += out["water_L"]
        self.total_cost += cost
        self.baseline_electricity_kWh += base["W_total_kW"]
        self.baseline_water_L += base["water_L"]
        self.baseline_cost += base_cost

        reward = (base_cost - cost) * self.reward_scale
        if self.reward_shaping:   # original hand-written arbitrage terms, kept only for ablation
            soc = self.tank.state_of_charge
            if soc > 0.98 or soc < 0.02:
                reward -= 0.5
            if beta < -0.2 and price <= PRICES["off_peak"]:
                reward += 3.0
            if beta > 0.2 and price >= PRICES["on_peak"]:
                reward += 6.0
            if price >= PRICES["on_peak"] and soc < 0.15:
                reward -= 10.0
            if beta < -0.1 and price >= PRICES["on_peak"]:
                reward -= 10.0

        self.current_step += 1
        done = self.current_step >= self.episode_hours
        info = {
            "electricity_savings_pct": (self.baseline_electricity_kWh - self.total_electricity_kWh) / max(1.0, self.baseline_electricity_kWh) * 100,
            "water_savings_pct": (self.baseline_water_L - self.total_water_L) / max(1.0, self.baseline_water_L) * 100,
            "cost_savings_pct": (self.baseline_cost - self.total_cost) / max(1e-9, self.baseline_cost) * 100,
            "tank_soc": self.tank.state_of_charge,
            "action_tank": beta,
            "price": price,
            "unmet_cooling_kWh": self.unmet_cooling_kWh,
            "Q_panel_kW": out["Q_panel_kW"],
            "COP": out["COP"],
        }
        return self._get_state(), reward, done, info

    def get_forecast(self, horizon: int = 6) -> np.ndarray:
        """Perfect forecast of [temp, rh, sky deficit, price] for the next `horizon` hours (normalised)."""
        rows = []
        for k in range(horizon):
            h = (self.current_hour + k) % self.n_hours
            rows.append([(self._wx["T_air_C"][h] - 10) / 40, self._wx["rh"][h],
                         self._sky_deficit[h] / 150.0, (self.get_electricity_price(self.current_hour + k) - 0.06) / 0.19])
        return np.clip(np.array(rows, dtype=np.float32), 0.0, 1.0)
