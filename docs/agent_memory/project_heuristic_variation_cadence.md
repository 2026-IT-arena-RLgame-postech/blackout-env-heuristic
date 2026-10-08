---
name: project-heuristic-variation-cadence
description: "Heuristic \"near\" parameter variation now resamples per absorption interval, not per full match — and what \"episode\" means in this codebase"
metadata: 
  node_type: memory
  type: project
  originSessionId: 12ffefa9-c57d-4c8e-9239-2ce090ada5e9
  modified: 2026-09-13T12:31:59.731Z
---

In this codebase "episode" (qmix_trainer.py docstring) means one absorption interval
(20s per GameBalanceConfig.asset), not the full 420s match — the RL trainer bootstraps/truncates n-step returns at
absorption boundaries (`buffer_done = done or absorption_fired`), since a full match is too
long a horizon.

Per user request (2026-09-13, before starting the training pipeline), heuristic behavior
policies used for BC/offline-RL data collection now resample their small "near" parameter
cloud (à la the existing `V4PolicyFamily`) at every absorption boundary instead of once per
match, for sampling diversity — while keeping the underlying strategy (`policy_id`, e.g.
which of v1-v16) fixed for the whole match ("기준 정책은 유지").

**Why:** more within-match behavioral diversity per unit of data collected, matching the
RL side's own absorption-granular episode framing, without churning which strategy
generated a trajectory (that would defeat BC/offline-RL provenance and diversity-by-design).

**How to apply:**
- `HeuristicPolicyMixture` (blackout_env/heuristics/mixture.py) now defaults
  `resample_each_absorption=True` — it's exclusively used for training-data collection
  (qmix_trainer, collect_heuristic_dataset*, parallel/actor.py), so this is safe as the
  unconditional default there. `policy_id` selection still only happens in `reset()`
  (called at real match boundaries); parameter resampling was split into
  `_resample_variation()`, called from `reset()` and again on every absorption-fired tick.
- 2026-09-16 fix: the absorption resample must *retune the running policy in place*
  (`StrategicHeuristic.retune()`, `V4PolicyFamily.adopt_sample()`), never rebuild it — the
  original rebuild wiped paths, roles, V10+ strategy modes and V8/V9 respec cooldowns (600
  ticks = 24s > the 20s interval) every boundary. `use_specialists` is structural and is
  fixed per match (drawn every time to keep the RNG stream, applied only on `reset()`).
- `V4PolicyFamily` (blackout_env/heuristics/v4_family.py) defaults
  `resample_each_absorption=False` (kept at the old once-per-match cadence) because
  `examples/benchmark_heuristics.py` and `examples/tournament_heuristics.py` construct it
  directly and read `candidate.current_sample.profile` *after* a whole match to attribute
  Elo/win-rate results to one profile — resampling mid-match there would silently break
  that per-match attribution. Only flip it to `True` for training-pipeline uses.
- Absorption-fired detection uses a tiny epsilon (`1e-6`) on `team_state[3]`
  (`absorption_time_left`), matching qmix_trainer's own detection; the pre-existing
  full-match detection on `team_state[2]` (`episode_time_left`) still uses the original
  `0.25` threshold.

See [[project_heuristic_numba_perf]] for related heuristic performance context.
