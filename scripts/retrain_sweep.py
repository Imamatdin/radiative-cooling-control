"""
Parallel retraining sweep on the corrected environment.

One job = (algorithm, city, seed, forecast mode: perfect | persistence | none, optional plant randomisation). Jobs run in separate processes (one CPU thread
each; the networks are small MLPs and the env step is CPU physics, so a GPU barely helps).
Every trained agent is scored with src.control.baselines.evaluate() on exactly the same test
episodes as the DP / MPC / schedule baselines in scripts/evaluate_baselines.py.

Train/test split uses the real calendar: test = episodes starting on Jan 1-29 and Jul 1-29 (local),
and no training episode's 48 h window touches any January or July hour.

Usage
  python scripts/retrain_sweep.py --shard 0/4                     # full sweep: 3 algos x 3 cities x 10 seeds x 3 forecast modes, quarter 0
  python scripts/retrain_sweep.py --smoke                         # 2-minute end-to-end check before paying for compute
Each finished job writes results/sweep/<job>.json and results/sweep/models/<job>.pt, so a killed
run resumes where it stopped (finished jobs are skipped).
"""
import argparse, json, os, sys, time, traceback
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))

WEATHER = {"phoenix": "data/weather/phoenix_az_tmy.csv",
           "houston": "data/weather/houston_tx_tmy.csv",
           "seattle": "data/weather/seattle_wa_tmy.csv"}
EPISODE_HOURS = 48
TEST_MONTHS = (1, 7)
OUT = os.path.join(ROOT, "results", "sweep")
REWARD_SCALE = 15.0


def calendar_months(n_hours):
    return pd.date_range("2021-01-01", periods=n_hours, freq="h").month.values


def test_starts(n_hours):
    days = pd.date_range("2021-01-01", periods=n_hours // 24, freq="D")
    return [i * 24 for i, d in enumerate(days) if d.month in TEST_MONTHS and d.day <= 29]


def train_starts(n_hours):
    test_hour = np.isin(calendar_months(n_hours), TEST_MONTHS)
    starts = []
    for h in range(0, n_hours - EPISODE_HOURS - 24):
        if not test_hour[h:h + EPISODE_HOURS].any():
            starts.append(h)
    return np.array(starts)


class TrainEnv:
    """Adapter with the reset(test=...) interface multi_city_training.train_agent expects.

    With several envs (domain randomisation over the plant), each episode runs on one drawn at random."""

    def __init__(self, envs, starts):
        self.envs = envs if isinstance(envs, list) else [envs]
        self.base_env, self.starts = self.envs[0], starts
        self.episode_hours = self.base_env.episode_hours
        self.state_dim, self.action_dim = self.base_env.state_dim, self.base_env.action_dim

    def reset(self, test=False):
        if test:
            raise RuntimeError("evaluation goes through src.control.baselines.evaluate")
        self.base_env = self.envs[np.random.randint(len(self.envs))]
        return self.base_env.reset(start_hour=int(np.random.choice(self.starts)))

    def step(self, a):
        return self.base_env.step(a)

    def get_forecast(self, horizon=6):
        return self.base_env.get_forecast(horizon)


FORECASTS = ("perfect", "persistence", "none")

# Plant-parameter ranges for domain-randomised training. Nominal values (PlantConfig defaults) sit inside each
# range. The held-out mismatch plants in scripts/eval_mismatch.py are separate draws, never the training pool.
DR_RANGES = {
    "solar_absorptance": (0.03, 0.10),     # film ageing / soiling raises absorbed sunlight
    "tower_approach_K": (4.0, 7.0),        # tower fouling / fill degradation
    "carnot_factor": (0.50, 0.65),         # chiller efficiency drift
    "emissivity": (0.88, 0.96),            # coating degradation
    "panel_area_scale": (0.70, 1.00),      # loops out of service / partial fouling (multiplies panel_area_m2)
}
DR_POOL = 8


def plant_draws(rng, n):
    from src.physics.condenser_loop import PlantConfig
    out = []
    for _ in range(n):
        kw = {k: float(rng.uniform(*r)) for k, r in DR_RANGES.items()}
        scale = kw.pop("panel_area_scale")
        cfg = PlantConfig(**kw)
        cfg.panel_area_m2 *= scale
        out.append(cfg)
    return out


def job_name(algo, city, seed, forecast, dr=False):
    suffix = {"perfect": "", "persistence": "_persist", "none": "_noforecast"}[forecast]
    return f"{algo}_{city}_s{seed}{suffix}" + ("_dr" if dr else "")


def run_job(spec):
    algo, city, seed, forecast, episodes, warmup, reward_scale, dr = spec
    name = job_name(algo, city, seed, forecast, dr)
    out_json = os.path.join(OUT, f"{name}.json")
    if os.path.exists(out_json):
        return name, "skipped (done)"
    try:
        import torch
        torch.set_num_threads(1)
        from multi_city_training import DDPGAgent, TD3Agent, SACAgent, train_agent, set_seed
        from src.environment.datacenter_env import DatacenterCoolingEnv, load_weather
        from src.control.baselines import AgentPolicy, evaluate

        set_seed(seed)
        use_fc = forecast != "none"
        fmode = "persistence" if forecast == "persistence" else "perfect"
        wdf = load_weather(os.path.join(ROOT, WEATHER[city]))
        if dr:
            cfgs = plant_draws(np.random.default_rng(10_000 + seed), DR_POOL)
            train_envs = [DatacenterCoolingEnv(wdf, episode_hours=EPISODE_HOURS, plant_cfg=c,
                                               reward_scale=reward_scale, forecast_mode=fmode) for c in cfgs]
        else:
            train_envs = [DatacenterCoolingEnv(wdf, episode_hours=EPISODE_HOURS, reward_scale=reward_scale,
                                               forecast_mode=fmode)]          # load noise on while training
        agent = {"DDPG": DDPGAgent, "TD3": TD3Agent, "SAC": SACAgent}[algo](
            state_dim=9, action_dim=2, use_forecast=use_fc, forecast_dim=24 if use_fc else 0)
        t0 = time.time()
        rewards, best = train_agent(TrainEnv(train_envs, train_starts(len(wdf))), agent, name,
                                    num_episodes=episodes, warmup=warmup, verbose=False)
        train_s = time.time() - t0
        os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
        agent.save(os.path.join(OUT, "models", f"{name}.pt"))

        starts = test_starts(len(wdf))
        eval_env = DatacenterCoolingEnv(wdf, episode_hours=EPISODE_HOURS, load_noise=False, forecast_mode=fmode)
        res = evaluate(eval_env, AgentPolicy(agent, name), starts)
        if forecast == "perfect":
            # Same agent deployed with the forecast a real plant would have.
            pe = evaluate(DatacenterCoolingEnv(wdf, episode_hours=EPISODE_HOURS, load_noise=False,
                                               forecast_mode="persistence"), AgentPolicy(agent, name), starts)
            res["deployed_with_persistence"] = {k: v for k, v in pe.items() if isinstance(v, (int, float))}
        res.update({"reward_scale": reward_scale, "algo": algo, "city": city, "seed": seed, "forecast": forecast,
                    "domain_randomised": dr, "episodes": episodes, "train_seconds": train_s,
                    "last100_reward": float(np.mean(rewards[-100:])), "reward_curve": [float(r) for r in rewards]})
        tmp = out_json + ".tmp"
        with open(tmp, "w") as f:
            json.dump(res, f)
        os.replace(tmp, out_json)
        return name, f"cost {res['cost_savings_pct']:.2f}%  water {res['water_savings_pct']:.1f}%  ({train_s/60:.1f} min)"
    except Exception:
        return name, "FAILED\n" + traceback.format_exc()


def build_jobs(args):
    seeds = list(range(args.seeds))
    jobs = []
    for fc in args.forecasts:
        for c in args.cities:
            for a in args.algos:
                for s in seeds:
                    jobs.append((a, c, s, fc, args.episodes, args.warmup, args.reward_scale, args.dr))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("GMN_CPU_LIMIT", 0)) or max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=3000)
    ap.add_argument("--reward-scale", type=float, default=REWARD_SCALE)
    ap.add_argument("--cities", nargs="+", default=list(WEATHER))
    ap.add_argument("--algos", nargs="+", default=["DDPG", "TD3", "SAC"])
    ap.add_argument("--forecasts", nargs="+", default=list(FORECASTS), choices=FORECASTS)
    ap.add_argument("--dr", action="store_true", help="domain-randomise the plant during training")
    ap.add_argument("--shard", type=str, default="0/1", help="i/n or i,j,k/n: run every n-th job starting at each listed i")
    ap.add_argument("--list", action="store_true", help="print this shard's job names and exit")
    ap.add_argument("--smoke", action="store_true", help="tiny run to verify the pipeline end to end")
    args = ap.parse_args()
    global OUT
    if args.smoke:
        OUT = os.path.join(ROOT, "results", "sweep_smoke")
        args.seeds, args.episodes, args.warmup, args.cities, args.algos = 1, 3, 100, ["houston"], ["SAC", "TD3"]
    sel, n = args.shard.split("/")
    n, all_jobs = int(n), build_jobs(args)
    jobs = [j for i in (int(x) for x in sel.split(",")) for j in all_jobs[i::n]]
    if args.list:
        print("\n".join(job_name(j[0], j[1], j[2], j[3], j[7]) for j in jobs))
        return
    os.makedirs(OUT, exist_ok=True)
    print(f"{len(jobs)} jobs (shard {args.shard}) on {args.workers} workers -> {OUT}", flush=True)
    t0 = time.time()
    with Pool(args.workers, maxtasksperchild=1) as pool:
        for k, (name, msg) in enumerate(pool.imap_unordered(run_job, jobs), 1):
            print(f"[{k}/{len(jobs)} {(time.time()-t0)/60:.0f}m] {name}: {msg}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
