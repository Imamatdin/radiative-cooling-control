"""
Parallel retraining sweep on the corrected environment.

One job = (algorithm, city, seed, forecast on/off). Jobs run in separate processes (one CPU thread
each; the networks are small MLPs and the env step is CPU physics, so a GPU barely helps).
Every trained agent is scored with src.control.baselines.evaluate() on exactly the same test
episodes as the DP / MPC / schedule baselines in scripts/evaluate_baselines.py.

Train/test split uses the real calendar: test = episodes starting on Jan 1-29 and Jul 1-29 (local),
and no training episode's 48 h window touches any January or July hour.

Usage
  python scripts/retrain_sweep.py --workers 96                    # full sweep: 3 algos x 3 cities x 10 seeds + forecast ablation
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
    """Adapter with the reset(test=...) interface multi_city_training.train_agent expects."""

    def __init__(self, env, starts):
        self.base_env, self.starts = env, starts
        self.episode_hours = env.episode_hours
        self.state_dim, self.action_dim = env.state_dim, env.action_dim

    def reset(self, test=False):
        if test:
            raise RuntimeError("evaluation goes through src.control.baselines.evaluate")
        return self.base_env.reset(start_hour=int(np.random.choice(self.starts)))

    def step(self, a):
        return self.base_env.step(a)

    def get_forecast(self, horizon=6):
        return self.base_env.get_forecast(horizon)


def job_name(algo, city, seed, forecast):
    return f"{algo}_{city}_s{seed}" + ("" if forecast else "_noforecast")


def run_job(spec):
    algo, city, seed, forecast, episodes, warmup = spec[:6]
    reward_scale = spec[6] if len(spec) > 6 else REWARD_SCALE
    name = job_name(algo, city, seed, forecast)
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
        wdf = load_weather(os.path.join(ROOT, WEATHER[city]))
        env = DatacenterCoolingEnv(wdf, episode_hours=EPISODE_HOURS, reward_scale=reward_scale)   # load noise on while training
        agent = {"DDPG": DDPGAgent, "TD3": TD3Agent, "SAC": SACAgent}[algo](
            state_dim=9, action_dim=2, use_forecast=forecast, forecast_dim=24 if forecast else 0)
        t0 = time.time()
        rewards, best = train_agent(TrainEnv(env, train_starts(len(wdf))), agent, name,
                                    num_episodes=episodes, warmup=warmup, verbose=False)
        train_s = time.time() - t0
        os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
        agent.save(os.path.join(OUT, "models", f"{name}.pt"))

        eval_env = DatacenterCoolingEnv(wdf, episode_hours=EPISODE_HOURS, load_noise=False)
        res = evaluate(eval_env, AgentPolicy(agent, name), test_starts(len(wdf)))
        res.update({"reward_scale": reward_scale, "algo": algo, "city": city, "seed": seed, "forecast": forecast,
                    "episodes": episodes, "train_seconds": train_s,
                    "last100_reward": float(np.mean(rewards[-100:])), "reward_curve": [float(r) for r in rewards]})
        tmp = out_json + ".tmp"
        with open(tmp, "w") as f:
            json.dump(res, f)
        os.replace(tmp, out_json)
        return name, f"cost {res['cost_savings_pct']:.2f}%  water {res['water_savings_pct']:.1f}%  ({train_s/60:.1f} min)"
    except Exception:
        return name, "FAILED\n" + traceback.format_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=3000)
    ap.add_argument("--cities", nargs="+", default=list(WEATHER))
    ap.add_argument("--algos", nargs="+", default=["DDPG", "TD3", "SAC"])
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny run to verify the pipeline end to end")
    args = ap.parse_args()
    global OUT
    if args.smoke:
        OUT = os.path.join(ROOT, "results", "sweep_smoke")
        args.seeds, args.episodes, args.warmup, args.cities, args.algos = 1, 3, 100, ["houston"], ["SAC", "TD3"]
    os.makedirs(OUT, exist_ok=True)
    seeds = list(range(args.seeds))
    jobs = [(a, c, s, True, args.episodes, args.warmup) for c in args.cities for a in args.algos for s in seeds]
    if not args.no_ablation:   # forecast ablation, all seeds (the original ran it on one seed)
        jobs += [(a, c, s, False, args.episodes, args.warmup) for c in args.cities for a in args.algos for s in seeds]
    print(f"{len(jobs)} jobs on {args.workers} workers -> {OUT}", flush=True)
    t0 = time.time()
    with Pool(args.workers, maxtasksperchild=1) as pool:
        for i, (name, msg) in enumerate(pool.imap_unordered(run_job, jobs), 1):
            print(f"[{i}/{len(jobs)} {(time.time()-t0)/60:.0f}m] {name}: {msg}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
