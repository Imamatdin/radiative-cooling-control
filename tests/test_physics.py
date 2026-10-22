"""Physics invariants the plant model must satisfy. Run: python -m pytest tests -q"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from src.physics.condenser_loop import (PlantConfig, plant_hour, panel_outlet, stagnation_T_K,
                                        panel_flux_W_m2, SIGMA)
from src.environment.datacenter_env import DatacenterCoolingEnv, load_weather, tou_price
from src.control.baselines import PlantTables, ConstantPolicy, MPCPolicy, TOUSchedulePolicy, evaluate

CFG = PlantConfig()
rng = np.random.default_rng(0)
N = 400
WX = {
    "T_air_C": rng.uniform(-15, 45, N),
    "rh": rng.uniform(0.05, 0.95, N),
    "L_down": rng.uniform(180, 450, N),
    "ghi": rng.uniform(0, 1000, N),
    "wind": rng.uniform(0, 8, N),
}


def test_condenser_loop_energy_balance_closes():
    for a in (0.0, 0.3, 1.0):
        out = plant_hour(rng.uniform(100, 1200, N), a, WX, CFG)
        assert np.all(np.abs(out["balance_residual_kW"]) < 1.0), "panel + tower must reject what the chiller rejects"
        assert np.all(out["Q_panel_kW"] >= -1e-6) and np.all(out["Q_tower_kW"] >= -1e-6)
        assert np.allclose(out["water_L"], out["Q_tower_kW"] * CFG.tower_water_L_per_kWh)


def test_panel_never_cools_fluid_below_stagnation_temperature():
    """Second-law bound: fluid cannot leave colder than the temperature where net panel flux is zero."""
    T_in = rng.uniform(15, 40, N)
    for mdot in (0.5, 5.0, 60.0):
        T_out, Q = panel_outlet(T_in, mdot, WX["T_air_C"], WX["L_down"], WX["ghi"], WX["wind"], CFG)
        T_stag = stagnation_T_K(WX["T_air_C"], WX["L_down"], WX["ghi"], WX["wind"], CFG) - 273.15
        assert np.all(T_out >= np.minimum(T_in, T_stag) - 1e-6)
        assert np.all(T_out <= T_in + 1e-9), "panels are valved off rather than heating the loop"
        assert np.all(Q >= -1e-6)


def test_stagnation_is_zero_flux_point():
    T_stag = stagnation_T_K(WX["T_air_C"], WX["L_down"], WX["ghi"], WX["wind"], CFG)
    q = panel_flux_W_m2(T_stag, WX["T_air_C"], WX["L_down"], WX["ghi"], WX["wind"], CFG)
    assert np.all(np.abs(q) < 0.1)


def test_no_sky_gain_when_radiation_disabled():
    dry = PlantConfig(radiative=False)
    T_out, _ = panel_outlet(30.0, 20.0, 35.0, 300.0, 0.0, 2.0, dry)
    assert float(T_out) == pytest.approx(30.0), "a dry cooler cannot reject heat to hotter air"


def test_tower_only_baseline_uses_no_panel_heat():
    out = plant_hour(800.0, 0.0, WX, CFG)
    assert np.all(out["Q_panel_kW"] == 0)
    assert np.all(out["W_pump_kW"] == 0)


def test_cop_rises_with_colder_condenser_water():
    from src.physics.condenser_loop import chiller_cop
    cops = chiller_cop(np.array([20.0, 25.0, 30.0, 35.0]), 0.6, CFG)
    assert np.all(np.diff(cops) <= 1e-9)


WEATHER = os.path.join(ROOT, "data", "weather", "phoenix_az_tmy.csv")


@pytest.fixture(scope="module")
def phx():
    return load_weather(WEATHER)


def test_weather_is_shifted_to_local_time(phx):
    """PVGIS rows are UTC. In Phoenix (UTC-7) solar noon must land near local hour 12, not 19."""
    ghi = phx["ghi"].values.reshape(-1, 24).mean(axis=0)
    assert 11 <= int(np.argmax(ghi)) <= 13
    assert ghi[0] == 0 and ghi[23] == 0


def test_sky_comes_from_measured_longwave_not_zero_cloud(phx):
    assert "cloud" not in phx.columns
    assert phx["L_down"].between(100, 500).all()


def test_tou_on_peak_is_local_afternoon():
    assert tou_price(15) == 0.25 and tou_price(3) == 0.06 and tou_price(10) == 0.12


def test_env_has_no_unmet_load_and_reward_is_cost_based(phx):
    env = DatacenterCoolingEnv(phx, episode_hours=24, load_noise=False)
    env.reset(start_hour=24 * 180, soc0=0.0)
    for _ in range(24):
        _, r, _, info = env.step([1.0, -1.0])   # charge every hour: worst case for chiller capacity
    assert info["unmet_cooling_kWh"] == 0.0
    env.reset(start_hour=24 * 180, soc0=0.5)
    _, r, _, _ = env.step([0.0, 0.0])
    assert r == pytest.approx(0.0, abs=1e-9), "tower-only must earn zero reward against the tower-only baseline"


def test_dp_oracle_is_at_least_as_good_as_every_other_controller(phx):
    env = DatacenterCoolingEnv(phx, episode_hours=48, load_noise=False)
    starts = [24 * 190, 24 * 200]
    tables = PlantTables(env, hours=np.concatenate([np.arange(s - 24, s + 72) for s in starts]))
    oracle = evaluate(env, MPCPolicy(tables, None, "perfect"), starts)["cost_savings_pct"]
    for p in (ConstantPolicy(0.0), ConstantPolicy(1.0), TOUSchedulePolicy(tables),
              MPCPolicy(tables, 24, "persistence")):
        assert evaluate(env, p, starts)["cost_savings_pct"] <= oracle + 0.2, p.name
