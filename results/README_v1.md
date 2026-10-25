# v1 results (Oct 25 2026): produced by agents with training bugs

`sweep_v1/`, `mismatch_v1/` and `tables_v1/` came from `scripts/multi_city_training.py` before commit 8428ccf.
That code had three bugs, all fixed since and covered by `tests/test_agents.py`:

1. The actor loss (all agents) and the DDPG TD target fed the critic raw tanh actions (alpha in [-1, 1]) while the
   critic was trained on environment-unit actions (alpha in [0, 1]). On a toy bandit with optimum alpha = 0.25 the
   old agents converge to alpha ~0.61, the value predicted by the unit mismatch ((0.25 + 1) / 2 = 0.625).
2. The 48 h episode time limit was stored as a terminal state, so stored cold had no value at episode end in
   training while evaluation credits it. Collapsed seeds sit at "tank discharging every hour".
3. "SACAgent" was a deterministic actor with no entropy term, not soft actor-critic.

Do not cite any RL number from these directories. The DP / MPC / schedule baselines in
`results/baselines_corrected.json` and the `base__*` mismatch files do not involve these agents and stay valid.
