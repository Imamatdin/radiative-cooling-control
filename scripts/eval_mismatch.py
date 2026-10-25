"""
Model-mismatch test: how each controller holds up when the real plant differs from the model it was built on.

MPC and the operator schedule plan with tables computed from the NOMINAL PlantConfig, then act on a plant whose
parameters have drifted. RL agents trained on the nominal plant, and agents trained with domain randomisation
(scripts/retrain_sweep.py --dr), act on the same drifted plant. Every number is a share of the exact optimum for
the drifted plant (DP with the TRUE model and a perfect forecast), on the same held-out Jan/Jul episodes.

The drifted plants below are fixed, named scenarios chosen before any result was seen. They are not the training
pool: the DR pool is 8 random draws per seed (retrain_sweep.plant_draws). Two scenarios sit outside the DR
ranges on purpose ("beyond_dr"), so DR agents get no credit for having trained on the test plant.

Usage
  python scripts/eval_mismatch.py --workers 28                        # all cities, all scenarios, all agents found
  python scripts/eval_mismatch.py --only baselines --cities houston
Each task writes results/mismatch/<task>.json; reruns skip finished tasks.
"""
import argparse, glob, json, os, re, sys, time, traceback
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from retrain_sweep import WEATHER, EPISODE_HOURS, test_starts  # noqa: E402

OUT = os.path.join(ROOT, "results", "mismatch")
MODELS = os.environ.get("MISMATCH_MODELS", os.path.join(ROOT, "results", "sweep", "models"))

# name -> (PlantConfig overrides, panel-area multiplier). Nominal: absorptance 0.05, approach 5 K,
# carnot 0.60, emissivity 0.95, area 5000 m2.
SCENARIOS = {
    "nominal":          ({}, 1.0),
    "aged_film":        ({"solar_absorptance": 0.09}, 1.0),
    "fouled_tower":     ({"tower_approach_K": 6.5}, 1.0),
    "chiller_drift":    ({"carnot_factor": 0.52}, 1.0),
    "degraded_coating": ({"emissivity": 0.89}, 1.0),
    "loops_offline":    ({}, 0.75),
    "combined_wear":    ({"solar_absorptance": 0.08, "tower_approach_K": 6.0, "carnot_factor": 0.55,
                          "emissivity": 0.91}, 0.85),
    "better_than_model": ({"solar_absorptance": 0.035, "tower_approach_K": 4.3, "carnot_factor": 0.63}, 1.0),
    "beyond_dr_film":   ({"solar_absorptance": 0.13}, 1.0),
    "beyond_dr_combined": ({"solar_absorptance": 0.12, "tower_approach_K": 7.5, "carnot_factor": 0.48,
                            "emissivity": 0.86}, 0.6),
}

NAME_RE = re.compile(r"^(DDPG|TD3|SAC)_(phoenix|houston|seattle)_s(\d+)(_persist|_noforecast)?(_dr)?$")


def plant_cfg(scenario):
    from src.physics.condenser_loop import PlantConfig
    kw, area = SCENARIOS[scenario]
    cfg = PlantConfig(**kw)
    cfg.panel_area_m2 *= area
    return cfg


def make_env(city, cfg, forecast_mode="perfect"):
    from src.environment.datacenter_env import DatacenterCoolingEnv, load_weather
    wdf = load_weather(os.path.join(ROOT, WEATHER[city]))
    return wdf, DatacenterCoolingEnv(wdf, episode_hours=EPISODE_HOURS, plant_cfg=cfg, load_noise=False,
                                     forecast_mode=forecast_mode)


def write(task, res):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"{task}.json")
    with open(p + ".tmp", "w") as f:
        json.dump(res, f)
    os.replace(p + ".tmp", p)


def baselines_task(spec):
    city, scenario = spec
    task = f"base__{city}__{scenario}"
    if os.path.exists(os.path.join(OUT, f"{task}.json")):
        return task, "skipped (done)"
    try:
        from src.physics.condenser_loop import PlantConfig
        from src.control.baselines import PlantTables, ConstantPolicy, TOUSchedulePolicy, MPCPolicy, evaluate
        t0 = time.time()
        wdf, true_env = make_env(city, plant_cfg(scenario))
        _, nom_env = make_env(city, PlantConfig())
        starts = test_starts(len(wdf))
        need = np.concatenate([np.arange(s - 24, s + EPISODE_HOURS + 24) for s in starts])
        nom_tables = PlantTables(nom_env, hours=need)      # what the controller believes
        true_tables = PlantTables(true_env, hours=need)    # the plant as it really is
        pols = {
            "dp_true_model": MPCPolicy(true_tables, None, "perfect"),
            "mpc_persist_nominal_model": MPCPolicy(nom_tables, 24, "persistence"),
            "mpc_persist_true_model": MPCPolicy(true_tables, 24, "persistence"),
            "tou_schedule_nominal_model": TOUSchedulePolicy(nom_tables),
            "field_always_on": ConstantPolicy(1.0, 0.0),
        }
        res = {k: evaluate(true_env, p, starts) for k, p in pols.items()}
        res["_meta"] = {"city": city, "scenario": scenario, "overrides": SCENARIOS[scenario][0],
                        "area_scale": SCENARIOS[scenario][1], "seconds": time.time() - t0}
        write(task, res)
        opt = res["dp_true_model"]["cost_savings_pct"]
        return task, f"DP {opt:.2f}%  MPC(nominal) {res['mpc_persist_nominal_model']['cost_savings_pct']:.2f}%  ({(time.time()-t0)/60:.1f} min)"
    except Exception:
        return task, "FAILED\n" + traceback.format_exc()


def agent_task(spec):
    city, scenario, model_path = spec
    name = os.path.basename(model_path)[:-3]
    task = f"rl__{city}__{scenario}__{name}"
    if os.path.exists(os.path.join(OUT, f"{task}.json")):
        return task, "skipped (done)"
    try:
        import torch
        torch.set_num_threads(1)
        from multi_city_training import DDPGAgent, TD3Agent, SACAgent
        from src.control.baselines import AgentPolicy, evaluate
        algo, c, seed, fc, dr = NAME_RE.match(name).groups()
        use_fc = fc != "_noforecast"
        mode = "persistence" if fc == "_persist" else "perfect"
        agent = {"DDPG": DDPGAgent, "TD3": TD3Agent, "SAC": SACAgent}[algo](
            state_dim=9, action_dim=2, use_forecast=use_fc, forecast_dim=24 if use_fc else 0)
        agent.load(model_path)
        wdf, env = make_env(city, plant_cfg(scenario), mode)
        res = evaluate(env, AgentPolicy(agent, name), test_starts(len(wdf)))
        res.update({"algo": algo, "city": city, "seed": int(seed), "forecast": mode if use_fc else "none",
                    "domain_randomised": bool(dr), "scenario": scenario, "agent": name})
        write(task, res)
        return task, f"{res['cost_savings_pct']:.2f}%"
    except Exception:
        return task, "FAILED\n" + traceback.format_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("GMN_CPU_LIMIT", 0)) or max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--cities", nargs="+", default=list(WEATHER))
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    ap.add_argument("--only", choices=["baselines", "rl"], default=None)
    ap.add_argument("--agent-glob", default="*_persist*.pt",
                    help="which trained agents to test (default: persistence-forecast agents, nominal and DR)")
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    tasks = []
    if args.only in (None, "baselines"):     # baselines first: the DP runs are the long poles
        tasks += [(baselines_task, (c, s)) for c in args.cities for s in args.scenarios]
    if args.only in (None, "rl"):
        for c in args.cities:
            for m in sorted(glob.glob(os.path.join(MODELS, args.agent_glob))):
                g = NAME_RE.match(os.path.basename(m)[:-3])
                if g and g.group(2) == c:
                    tasks += [(agent_task, (c, s, m)) for s in args.scenarios]
    i, n = (int(x) for x in args.shard.split("/"))
    tasks = tasks[i::n]
    print(f"{len(tasks)} tasks on {args.workers} workers -> {OUT}", flush=True)
    t0 = time.time()
    with Pool(args.workers, maxtasksperchild=4) as pool:
        for k, (task, msg) in enumerate(pool.imap_unordered(_star, tasks), 1):
            print(f"[{k}/{len(tasks)} {(time.time()-t0)/60:.0f}m] {task}: {msg}", flush=True)
    print("done", flush=True)


def _star(fs):
    return fs[0](fs[1])


if __name__ == "__main__":
    main()
