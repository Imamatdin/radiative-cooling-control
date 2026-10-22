"""
Control baselines for the datacenter cooling environment.

Structure of the problem (why exact optimisation is cheap here):
- The panel split alpha does not change any state that carries over to the next hour. Given the
  chiller load in an hour, the best alpha is a static per-hour minimisation.
- The only dynamic state is the tank's stored cold E (kWh). So the full problem is a 1-D dynamic
  program over E, with per-hour stage costs that are themselves minimised over alpha.

Controllers
- ConstantPolicy          : fixed (alpha, beta), e.g. tower-only (0,0) or panels-always-on (1,0).
- TOUSchedulePolicy       : what plant operators actually run: charge the tank off-peak, discharge on-peak,
                            panels on whenever they reduce cost this hour (myopic alpha).
- MPCPolicy               : receding-horizon DP over the tank with the plant model in the loop.
      forecast='perfect'      -> perfect weather over the horizon
      forecast='persistence'  -> weather for hour t+k forecast as the observed weather at t+k-24
                                 (24 h persistence; current hour is observed)
      horizon=None            -> plan to episode end with perfect forecast = DP oracle (optimal bound)

All controllers credit/charge the change in stored cold at the same value (VALUE_PER_KWH_COLD) so that
draining a tank that started full is not counted as free savings. evaluate() applies the same correction.
"""

import numpy as np

from src.environment.datacenter_env import tou_price, it_cooling_load, PRICES
from src.physics.condenser_loop import plant_hour

# Value of 1 kWh of stored cold at episode end: cost of producing it off-peak at a typical COP.
VALUE_PER_KWH_COLD = PRICES["off_peak"] / 6.0

ALPHA_GRID = np.linspace(0.0, 1.0, 11)


class PlantTables:
    """W_total(hour_weather, q, alpha) and water(hour_weather, q, alpha) for every weather hour, computed once."""

    def __init__(self, env, q_step_kW=25.0, chunk=96, hours=None):
        self.env = env
        cap = env.cfg.chiller_capacity_kW
        self.q_grid = np.arange(0.0, cap + 1e-9, q_step_kW)
        self.q_step = q_step_kW
        n, nq, na = env.n_hours, len(self.q_grid), len(ALPHA_GRID)
        self.W = np.empty((n, nq, na), np.float32)
        self.water = np.empty((n, nq, na), np.float32)
        q = self.q_grid[None, :, None]
        a = ALPHA_GRID[None, None, :]
        need = np.arange(n) if hours is None else np.unique(np.asarray(hours) % n)
        self.W[:] = np.nan
        self.water[:] = np.nan
        for s in range(0, len(need), chunk):
            idx = need[s:s + chunk]
            wx = {k: v[idx][:, None, None] for k, v in env._wx.items()}
            out = plant_hour(q, a, wx, env.cfg)
            self.W[idx] = out["W_total_kW"]
            self.water[idx] = out["water_L"]

    def stage_cost(self, weather_hours, target_hours):
        """Best-alpha cost table C[k, q] and argmin alpha A[k, q] for each horizon step."""
        price = tou_price(np.asarray(target_hours) % 24)[:, None, None]
        if np.isnan(self.W[weather_hours, 0, 0]).any():
            raise ValueError("PlantTables built without some requested hours; pass hours= covering the evaluation window")
        c = self.W[weather_hours] * price + self.water[weather_hours] * self.env.water_price
        j = np.argmin(c, axis=2)
        return np.take_along_axis(c, j[..., None], 2)[..., 0], ALPHA_GRID[j]


def _interp_q(table_row, q, q_step):
    """Linear interpolation of a per-q table row at (array) q."""
    x = np.clip(q / q_step, 0, table_row.shape[-1] - 1 - 1e-9)
    i = np.floor(x).astype(int)
    f = x - i
    return table_row[i] * (1 - f) + table_row[np.minimum(i + 1, table_row.shape[-1] - 1)] * f


def plan_tank(C, A, loads, E0, Emax, cap, q_step, P_max=500.0, nE=49, nP=41):
    """
    Backward DP over stored cold E for one horizon. Returns (alpha, P) for the first step,
    where P > 0 discharges the tank and P < 0 charges it [kW, 1-hour step].
    """
    H = C.shape[0]
    E_grid = np.linspace(0.0, Emax, nE)
    P_grid = np.linspace(-P_max, P_max, nP)
    V = -VALUE_PER_KWH_COLD * E_grid               # terminal: stored cold has value
    policy_P = np.zeros((H, nE))
    for k in range(H - 1, -1, -1):
        E = E_grid[:, None]
        P = P_grid[None, :]
        dis = np.clip(np.minimum(P, np.minimum(E, loads[k])), 0, None)
        chg = np.clip(np.minimum(-P, np.minimum(Emax - E, max(0.0, cap - loads[k]))), 0, None)
        E_next = E - dis + chg
        q = loads[k] - dis + chg
        stage = _interp_q(C[k], q, q_step)
        V_next = np.interp(E_next.ravel(), E_grid, V).reshape(E_next.shape)
        total = stage + V_next
        j = np.argmin(total, axis=1)
        V = total[np.arange(nE), j]
        policy_P[k] = P_grid[j]
    P0 = float(np.interp(E0, E_grid, policy_P[0]))
    dis0 = max(0.0, min(P0, E0, loads[0]))
    chg0 = max(0.0, min(-P0, Emax - E0, max(0.0, cap - loads[0])))
    q0 = loads[0] - dis0 + chg0
    a0 = float(A[0][int(np.clip(round(q0 / q_step), 0, A.shape[1] - 1))])
    return a0, P0


class ConstantPolicy:
    def __init__(self, alpha, beta=0.0, name=None):
        self.a, self.b = alpha, beta
        self.name = name or f"Fixed alpha={alpha}"

    def reset(self, env):
        pass

    def act(self, env):
        return np.array([self.a, self.b])


class TOUSchedulePolicy:
    """Charge 22:00-06:00 local, discharge 14:00-19:00 local, myopic best alpha each hour."""

    name = "Operator TOU schedule"

    def __init__(self, tables):
        self.T = tables

    def reset(self, env):
        pass

    def act(self, env):
        h = env.current_hour
        hod = h % 24
        beta = -1.0 if (hod >= 22 or hod < 6) else (1.0 if 14 <= hod < 19 else 0.0)
        load = float(it_cooling_load(hod))
        q = load - (min(beta * env.tank_power_kW, env.tank_stored_kWh) if beta > 0 else 0.0) \
            + (min(-beta * env.tank_power_kW, env.tank_capacity_kWh - env.tank_stored_kWh, env.cfg.chiller_capacity_kW - load) if beta < 0 else 0.0)
        _, A = self.T.stage_cost(np.array([h % env.n_hours]), np.array([h]))
        alpha = A[0][int(np.clip(round(q / self.T.q_step), 0, A.shape[1] - 1))]
        return np.array([alpha, beta])


class MPCPolicy:
    def __init__(self, tables, horizon=24, forecast="perfect", name=None):
        self.T = tables
        self.H = horizon
        self.forecast = forecast
        self.name = name or (f"MPC H={horizon}h ({forecast} forecast)" if horizon else "DP oracle (perfect forecast, full episode)")

    def reset(self, env):
        self.end = env.start_hour + env.episode_hours

    def act(self, env):
        t = env.current_hour
        H = (self.end - t) if self.H is None else self.H
        target = t + np.arange(H)
        if self.forecast == "perfect":
            wh = target % env.n_hours
        elif self.forecast == "persistence":
            wh = np.where(np.arange(H) == 0, t, target - 24) % env.n_hours
        else:
            raise ValueError(self.forecast)
        C, A = self.T.stage_cost(wh, target)
        loads = it_cooling_load(target % 24)
        a, P = plan_tank(C, A, loads, env.tank_stored_kWh, env.tank_capacity_kWh,
                         env.cfg.chiller_capacity_kW, self.T.q_step, env.tank_power_kW)
        return np.array([a, np.clip(P / env.tank_power_kW, -1, 1)])


class AgentPolicy:
    """Adapter so trained RL agents (anything with select_action(state, forecast, noise)) run through evaluate()."""

    def __init__(self, agent, name):
        self.agent, self.name = agent, name

    def reset(self, env):
        pass

    def act(self, env):
        fc = env.get_forecast() if getattr(self.agent, "use_forecast", False) else None
        return np.asarray(self.agent.select_action(env._get_state(), fc, noise=0), float)


def run_episode(env, policy, start_hour, soc0=0.5):
    env.reset(start_hour=start_hour, soc0=soc0)
    policy.reset(env)
    done = False
    while not done:
        _, _, done, info = env.step(policy.act(env))
    tank_credit = VALUE_PER_KWH_COLD * (env.tank_stored_kWh - env.tank_start_kWh)
    return {
        "cost": env.total_cost - tank_credit,
        "base_cost": env.baseline_cost,
        "elec_kWh": env.total_electricity_kWh, "base_elec_kWh": env.baseline_electricity_kWh,
        "water_L": env.total_water_L, "base_water_L": env.baseline_water_L,
        "unmet_kWh": info["unmet_cooling_kWh"],
    }


def evaluate(env, policy, start_hours, soc0=0.5):
    rs = [run_episode(env, policy, s, soc0) for s in start_hours]
    agg = {k: float(np.sum([r[k] for r in rs])) for k in rs[0]}
    per_ep_cost_sav = [(r["base_cost"] - r["cost"]) / r["base_cost"] * 100 for r in rs]
    return {
        "cost_savings_pct": (agg["base_cost"] - agg["cost"]) / agg["base_cost"] * 100,
        "cost_savings_pct_std": float(np.std(per_ep_cost_sav)),
        "elec_savings_pct": (agg["base_elec_kWh"] - agg["elec_kWh"]) / agg["base_elec_kWh"] * 100,
        "water_savings_pct": (agg["base_water_L"] - agg["water_L"]) / agg["base_water_L"] * 100,
        "unmet_kWh": agg["unmet_kWh"],
        "cost_saved_usd": agg["base_cost"] - agg["cost"],
        "episodes": len(rs),
    }
