"""
Evaluate control baselines on the held-out months (January and July, local time).

Reports cost / electricity / water savings vs tower-only, and how much of the optimal (DP oracle)
cost saving each controller captures. RL agents should be evaluated through the same
src.control.baselines.evaluate() so every number in the paper comes from one harness.

Usage:  python scripts/evaluate_baselines.py [--cities phoenix houston seattle] [--out results/baselines.json]
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from src.environment.datacenter_env import DatacenterCoolingEnv, load_weather
from src.physics.condenser_loop import PlantConfig
from src.control.baselines import (PlantTables, ConstantPolicy, TOUSchedulePolicy, MPCPolicy, evaluate)

WEATHER = {"phoenix": "data/weather/phoenix_az_tmy.csv",
           "houston": "data/weather/houston_tx_tmy.csv",
           "seattle": "data/weather/seattle_wa_tmy.csv"}
TEST_MONTHS = (1, 7)
EPISODE_HOURS = 48


def test_starts(n_hours, months=TEST_MONTHS):
    """Local-midnight starts on days 1-29 of each test month (TMY has no leap day)."""
    days = pd.date_range("2021-01-01", periods=n_hours // 24, freq="D")
    return [i * 24 for i, d in enumerate(days) if d.month in months and d.day <= 29]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=list(WEATHER))
    ap.add_argument("--out", default="results/baselines_corrected.json")
    args = ap.parse_args()
    os.chdir(ROOT)
    results = {}
    for city in args.cities:
        wdf = load_weather(WEATHER[city])
        starts = test_starts(len(wdf))
        rows = {}
        for plant_name, cfg in [("panels", PlantConfig()), ("dry_cooler_same_area", PlantConfig(radiative=False))]:
            env = DatacenterCoolingEnv(wdf, episode_hours=EPISODE_HOURS, plant_cfg=cfg, load_noise=False)
            t0 = time.time()
            need = np.concatenate([np.arange(s0 - 24, s0 + EPISODE_HOURS + 24) for s0 in starts])
            tables = PlantTables(env, hours=need)
            policies = [ConstantPolicy(0.0, 0.0, "Tower only"),
                        ConstantPolicy(1.0, 0.0, "Field always on, no storage"),
                        TOUSchedulePolicy(tables),
                        MPCPolicy(tables, 24, "persistence"),
                        MPCPolicy(tables, 24, "perfect"),
                        MPCPolicy(tables, None, "perfect")]
            if plant_name == "dry_cooler_same_area":
                policies = policies[1:3] + policies[-1:]
            for p in policies:
                r = evaluate(env, p, starts)
                rows[f"{plant_name} | {p.name}"] = r
            print(f"[{city}] {plant_name}: {len(policies)} controllers x {len(starts)} episodes in {time.time()-t0:.0f}s", flush=True)
        oracle = rows["panels | DP oracle (perfect forecast, full episode)"]["cost_savings_pct"]
        for r in rows.values():
            r["pct_of_optimal_cost_saving"] = 100 * r["cost_savings_pct"] / oracle if oracle > 0 else float("nan")
        results[city] = rows
        print(f"\n{city.upper()}  (test: Jan+Jul local, {len(starts)} x {EPISODE_HOURS} h, tank starts at 50 % SOC)")
        print(f"{'controller':62s} {'cost%':>7s} {'elec%':>7s} {'water%':>7s} {'%opt':>6s} {'unmet':>6s}")
        for k, r in rows.items():
            print(f"{k:62s} {r['cost_savings_pct']:7.2f} {r['elec_savings_pct']:7.2f} {r['water_savings_pct']:7.2f} "
                  f"{r['pct_of_optimal_cost_saving']:6.1f} {r['unmet_kWh']:6.1f}")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:          # saved after every city so a killed run keeps its results
            json.dump(results, f, indent=2)
        print(f"wrote {args.out} ({', '.join(results)})", flush=True)


if __name__ == "__main__":
    main()
