# PPO league arm B (setup)

Branch `worktree-agent-abc677424de1495fd` (WSL: `league-b`). Recipe = `ppo_r2.sh ARM=reg` + the flags below (all default off; flags off is
bit-identical to the old trainer, checked by hashing batches and final weights over 3 iterations vs `feat/official-obs`).
Code: `blackout_env/rl/league.py` (pure logic), `PPOTrainer` in `blackout_env/rl/ppo.py`, `examples/rl_train.py` (`--save-every`, `--resume`).
Launcher `reports/rl/ppo_league_b.sh`; tests `tests/test_league.py`.

* `adv_center_opp`: per opponent, subtract the batch-mean of the raw GAE advantage over its rows if it has >= `adv_center_min_games`=10 distinct
  games (finished or active) in the batch; then the usual global std scaling (std of the centred advantage). The critic target `ret` is not centred.
* `league`: gate = sum_i w_i * opp_score_i over the fixed pool, w = base weight x PFSP factor renormalised over the fixed pool (the fixed pool's
  current sampling distribution); opp_score = per-opponent EMA (rate `pfsp_ema`) of the learner's result (win 1, draw 1/2) on games that ended.
  Snapshot when gate >= 0.85, >= 100 its since the last snapshot, >= 150 its since start, < `league_max` snapshots. The learner is saved as
  `<run>/snap<k>_it<N>.pt` (a normal policy checkpoint) and appended to `opps` as a NetPolicy; 2 critic-only iterations follow (`warm_until`).
* Share ramp: snapshot shares 5 % -> 10 % -> 20 % of trunk games (`league_steps`); a step needs >= 20 its since the last one (or since creation),
  >= `league_ramp_min_games` finished games of that snapshot (lifetime), adv-std ratio <= 1.5 (pooled std of the advantage fed to the policy over
  the snapshot's rows in the last 20 its / same for all fixed-pool rows) and EV_mc >= 0.4 (EV_mc = 1 - Var(R-V)/Var(R) over all rows of the
  snapshot's finished games, V = critic at collection time, R = the game's final reward). Fixed pool keeps >= 40 % (snapshot shares are scaled
  down together if they would exceed 60 %); PFSP reweights the fixed pool within the remaining mass; snapshots are not PFSP-reweighted.
* Per-iteration log keys: `opp_w` (sampling weights after this iteration), `opp_share` (games per opponent in the batch), `opp_score`, `opp_adv_mean`
  (raw), `opp_adv_std` (used), `opp_ev_mc`, `opp_fin`, `gate_score`, `n_snaps`, `snap_levels`, `snap_games`, `snap_ev_mc`, `league_events`
  (snapshot / ramp_step records), `adv_std` (scale used) and `adv_std_raw`.
* Resume: `latest.pt` every 10 its (atomic) holds policy, critic, optimisers, scalers, iteration, PFSP scores, finished-game sums, rng, ramp state;
  snapshot weights are the `snap*.pt` files. In-flight trunk games restart from fresh seeds. Run `RESUME=1` with the same env as the original launch.

## WSL laptop run (RTX 3060 6 GB, 12 cores / 24 GB RAM)
Run dir `~/blackout-env-league/checkpoints/rl/league_b` (worktree of the WSL mirror, branch `league-b`), launcher `reports/rl/ppo_league_b.sh`
(`MAXDEC=33327244`; resume: `RESUME=1 MAXDEC=33327244 reports/rl/ppo_league_b.sh`). r2 recipe unchanged (games 256, segment 128, minibatch 1024,
fp16/compile/critic_warmup 10, eval every 150 its x 16 seeds) plus the memory changes needed for 6 GB, all numerics-neutral up to fp rounding:
* `graph_shared_pool` (all CUDA-graph forwards share one pool), `league_snap_graphs` (snapshots use graphs: eager snapshots cost +17 s/it),
* `critic_read_batch=1024` (the 4096-row eager critic read peaked at ~2.4 GB), `update_micro=512` (gradient accumulation of 2 x 512 inside each
  1024 minibatch; update-phase peak 3.5 -> 2.2 GB; test shows identical gradients), `cuda_mem_fraction=0.70` (hard stop instead of a full GPU).
Smoke (forced snapshots every 3 its, 4 snapshots, all at max share): nvidia-smi peak 4.27 GB, torch allocated peak 2.3 GB, 18 s/it on
critic-only iterations and ~32 s/it with policy update; heuristic workers 6, OMP 2. Evals (16 seeds x 2 sides x 4 opponents) take ~25 min each.
Earlier smoke variants that filled the GPU to 5.9-6.0 GB (and two jobs sharing the GPU for ~5 min) logged 300 `nvlddmkm` id-13 events; none since
the memory was cut.
