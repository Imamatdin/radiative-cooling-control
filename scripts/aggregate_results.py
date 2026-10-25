"""
Aggregate the retraining sweep and the model-mismatch test into the paper's tables.

Inputs (all produced by the scripts named; nothing hand-typed):
  results/baselines_corrected.json      scripts/evaluate_baselines.py
  results/sweep/*.json                  scripts/retrain_sweep.py
  results/mismatch/*.json               scripts/eval_mismatch.py
Outputs:
  results/tables/sweep_summary.csv      per city x algo x forecast x DR: mean, std, 95% bootstrap CI, % of optimum
  results/tables/mismatch_summary.csv   per city x scenario x controller: % of the drifted plant's optimum
  results/tables/summary.md             the same, readable

A run is flagged "collapsed" when its test score equals the constant corner policy (alpha=1, beta=+1:
field always on, tank discharging every hour) to within 0.01 points. Collapsed runs are kept in every
mean (dropping failed seeds would flatter RL); the collapse rate is reported next to it.
"""
import csv, glob, json, os, sys
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
RES = os.path.join(ROOT, "results")
OUTD = os.path.join(RES, "tables")
CITIES = ["phoenix", "houston", "seattle"]
ORACLE = "panels | DP oracle (perfect forecast, full episode)"


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def corner_scores(cache=os.path.join(RES, "tables", "corner_policy.json")):
    if os.path.exists(cache):
        return json.load(open(cache))
    from src.environment.datacenter_env import DatacenterCoolingEnv, load_weather
    from src.control.baselines import ConstantPolicy, evaluate
    from retrain_sweep import WEATHER, EPISODE_HOURS, test_starts
    out = {}
    for c in CITIES:
        w = load_weather(os.path.join(ROOT, WEATHER[c]))
        env = DatacenterCoolingEnv(w, episode_hours=EPISODE_HOURS, load_noise=False)
        out[c] = evaluate(env, ConstantPolicy(1.0, 1.0), test_starts(len(w)))["cost_savings_pct"]
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(out, open(cache, "w"))
    return out


def sweep_table(base, corner):
    runs = [json.load(open(f)) for f in glob.glob(os.path.join(RES, "sweep", "*.json"))]
    runs = [r for r in runs if "cost_savings_pct" in r]
    groups = defaultdict(list)
    for r in runs:
        groups[(r["city"], r["algo"], r["forecast"], bool(r.get("domain_randomised", False)))].append(r)
    rows = []
    for (city, algo, fc, dr), rs in sorted(groups.items()):
        opt = base[city][ORACLE]["cost_savings_pct"]
        cost = [r["cost_savings_pct"] for r in rs]
        lo, hi = boot_ci(cost)
        row = {"city": city, "algo": algo, "forecast": fc, "dr": dr, "n": len(rs),
               "cost_mean": np.mean(cost), "cost_std": np.std(cost, ddof=1) if len(cost) > 1 else float("nan"),
               "ci_lo": lo, "ci_hi": hi, "pct_opt": 100 * np.mean(cost) / opt,
               "pct_opt_best_seed": 100 * max(cost) / opt,
               "water_mean": np.mean([r["water_savings_pct"] for r in rs]),
               "collapsed": sum(abs(c - corner[city]) < 0.01 for c in cost)}
        dep = [r["deployed_with_persistence"]["cost_savings_pct"] for r in rs if "deployed_with_persistence" in r]
        row["pct_opt_deployed_persist"] = 100 * np.mean(dep) / opt if dep else float("nan")
        rows.append(row)
    return rows


def mismatch_table():
    files = glob.glob(os.path.join(RES, "mismatch", "*.json"))
    base, rl = {}, defaultdict(list)
    for f in files:
        r = json.load(open(f))
        name = os.path.basename(f)[:-5]
        if name.startswith("base__"):
            _, city, scen = name.split("__")
            base[(city, scen)] = r
        else:
            rl[(r["city"], r["scenario"], r["algo"], r["domain_randomised"])].append(r["cost_savings_pct"])
    rows = []
    for (city, scen), b in sorted(base.items()):
        opt = b["dp_true_model"]["cost_savings_pct"]
        for k in ("mpc_persist_nominal_model", "mpc_persist_true_model", "tou_schedule_nominal_model",
                  "field_always_on"):
            rows.append({"city": city, "scenario": scen, "controller": k, "n": 1,
                         "pct_opt": 100 * b[k]["cost_savings_pct"] / opt, "ci_lo": float("nan"),
                         "ci_hi": float("nan"), "opt_cost_saving": opt})
        for algo in ("DDPG", "TD3", "SAC"):
            for dr in (False, True):
                xs = rl.get((city, scen, algo, dr))
                if not xs:
                    continue
                p = [100 * x / opt for x in xs]
                lo, hi = boot_ci(p)
                rows.append({"city": city, "scenario": scen, "controller": f"{algo}{' DR' if dr else ''} (persistence fc)",
                             "n": len(p), "pct_opt": float(np.mean(p)), "ci_lo": lo, "ci_hi": hi,
                             "opt_cost_saving": opt})
    return rows


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    os.makedirs(OUTD, exist_ok=True)
    base = json.load(open(os.path.join(RES, "baselines_corrected.json")))
    corner = corner_scores()
    s = sweep_table(base, corner)
    m = mismatch_table()
    write_csv(s, os.path.join(OUTD, "sweep_summary.csv"))
    write_csv(m, os.path.join(OUTD, "mismatch_summary.csv"))
    L = ["# Sweep summary (test: Jan+Jul, cost saving vs tower-only; % of DP optimum)\n"]
    for city in CITIES:
        b = base[city]
        L.append(f"\n## {city}  (DP optimum {b[ORACLE]['cost_savings_pct']:.2f}%; operator schedule "
                 f"{b['panels | Operator TOU schedule']['pct_of_optimal_cost_saving']:.1f}%; MPC persistence "
                 f"{b['panels | MPC H=24h (persistence forecast)']['pct_of_optimal_cost_saving']:.1f}%)\n")
        L.append("| algo | forecast | DR | n | cost % (mean ± sd) | 95% CI | % opt | best seed % opt | deployed w/ persistence % opt | collapsed |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in [r for r in s if r["city"] == city]:
            dep = r["pct_opt_deployed_persist"]
            dep_s = "" if np.isnan(dep) else f"{dep:.1f}"
            L.append(f"| {r['algo']} | {r['forecast']} | {'yes' if r['dr'] else ''} | {r['n']} | "
                     f"{r['cost_mean']:.2f} ± {r['cost_std']:.2f} | {r['ci_lo']:.2f}–{r['ci_hi']:.2f} | "
                     f"{r['pct_opt']:.1f} | {r['pct_opt_best_seed']:.1f} | {dep_s} | {r['collapsed']}/{r['n']} |")
    if m:
        L.append("\n# Model mismatch (% of the drifted plant's own optimum; RL = mean over seeds, 95% CI)\n")
        for city in CITIES:
            scens = sorted({r["scenario"] for r in m if r["city"] == city})
            ctrls = list(dict.fromkeys(r["controller"] for r in m if r["city"] == city))
            if not scens:
                continue
            L.append(f"\n## {city}\n")
            L.append("| controller | " + " | ".join(scens) + " |")
            L.append("|---|" + "---|" * len(scens))
            idx = {(r["scenario"], r["controller"]): r for r in m if r["city"] == city}
            for c in ctrls:
                cells = []
                for sc in scens:
                    r = idx.get((sc, c))
                    cells.append("" if r is None else (f"{r['pct_opt']:.1f}" if r["n"] == 1 else
                                                       f"{r['pct_opt']:.1f} ({r['ci_lo']:.0f}–{r['ci_hi']:.0f}, n={r['n']})"))
                L.append(f"| {c} | " + " | ".join(cells) + " |")
    open(os.path.join(OUTD, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
