# PPO iteration profile, current recipe (2026-10-30)

poscat_gpu2 GPU 2 (2080 Ti), server load ~3. `reports/rl/q_prof.sh` → `examples/ppo_profile.py --iters 5`, L1 pool
(9 B + V33 + F2 + F3: 10 net opponents, 3 heuristic), `games=256 segment=128` (32.8k views/iter), heur_workers 6,
update_fp16 + critic_fp16 + update_compile, update_micro 512. Raw: `reports/rl/prof/{nowrap,wrap}.jsonl`
(nowrap = wall phases + JointSeats split timers; wrap = per-policy synced timers). Steady state = iters 1–4 (iter 0 = compile/warm-up).

## Where an iteration goes (nowrap, mean of iters 1–4)
| phase | s | share |
|---|---|---|
| trunk rollout | 9.5 | 43 % |
| ├ network forwards (learner + 10 net opponents, synced) | **8.3** | 38 % |
| ├ heuristic start (ship states to workers) | 0.75 | 3 % |
| ├ heuristic wait after the forwards | **0.07** | 0.3 % |
| └ render + sim step | 0.35 | 2 % |
| leaf (critic values over 32.8k views + bootstrap) | 1.4 | 6 % |
| update (critic epochs ≈ 4.2 + policy old/ref pass + 2 epochs ≈ 6.9) | 11.1 | 51 % |
| **total** | **22.0** | |

Per policy (wrap run): learner 2.2 s, each net opponent 0.63–0.73 s (10 × ≈ 0.67 = 6.7 s). Each net opponent plays ~1/10
of 256 games, i.e. ~25-row forwards × 2 teams × 128 steps: **latency-bound small forwards**, not compute.

## What the two speed-up ideas can save
1. **Split the games in two halves and alternate (heuristics on one half while the GPU runs the other):** the heuristics are
   *already* fully hidden behind the forwards (`overlap_heur`: wait 0.07 s/iter). The only CPU work left on the critical path is
   render + step (0.35 s) and heuristic start (0.75 s) → at most **~1 s/iter (≈ 5 %)**, and smaller forward batches would cost
   more launch latency than that. **Not worth it.**
2. **Async actor-learner (rollout k+1 while update k runs):** critical path becomes max(trunk + leaf ≈ 10.9, update ≈ 11.1)
   instead of the sum (22.0) → **up to ~2× (≈ 11–12 s/iter)**, minus GIL/launch contention.
   - On **2 GPUs** (actor on one, learner on the other) contention is only the Python GIL: rollout forwards are graph replays
     (little Python), the compiled fp16 update is kernel-heavy → expect most of the 2×.
   - Also worth trying on **1 GPU**: the rollout's forwards are latency-bound tiny batches that leave the GPU mostly idle, so
     the update's large kernels can fill the gaps. Expected less than 2× but testable on GPU2 alone (no second free GPU needed).
   - Cost: 1-iteration policy lag. The PPO ratio must use the *behaviour* policy's log-probs (the weights that played the
     rollout), not a recomputed pass with the current net; values/GAE from the actor's critic copy are 1 iteration stale.
3. **New, from the profile — cheaper net-opponent forwards** (8.3 s of 9.5 s trunk): 11 policies × 2 team calls per step.
   Options: run the opponents' forwards on separate CUDA streams (independent small graphs overlap on the GPU), or one forward
   per opponent for both teams (`joint=true`, not bit-identical). Potential trunk 9.5 → ~5 s (−20 % of the iteration), and it
   stacks with idea 2.

## Async actor-learner, measured (`async_collect`, ddac4a1; `reports/rl/q_async_bench.sh`, `examples/ppo_async_bench.py`)
Same recipe/pool as above, 7 iterations each, steady state = iters 1–6. Raw: `reports/rl/prof/async_{sync,a1,a2}.jsonl`.

| mode | wall s/iter | speed-up | trunk (in the actor) | update | learner waits for actor |
|---|---|---|---|---|---|
| sync (baseline) | 22.1 | 1.0× | 9.7 | 11.1 | – |
| async, 1 GPU (2nd CUDA stream) | 18.9 | **1.17×** | 17–22 (contention) | 12.4 | 6.4 |
| async, 2 GPUs (learner GPU2, actor GPU0) | **12.9** | **1.71×** | 10.5 | 12.3 | 0–2.5 |

- 1 GPU: the rollout's latency-bound replays queue behind the update's kernels (and the GIL), the trunk nearly doubles → small gain.
- **2 GPUs: ~1.7×**, close to the max(trunk + leaf, update) bound; the update (≈ 12 s) is now the critical path, so further rollout
  speed-ups (stream-parallel opponent forwards) would only help together with a faster update.
- KL / clipfrac in the async runs stay in the sync range (KL 0.002–0.003 after 6 its, clipfrac 0.013–0.016 vs 0.008–0.011):
  the 1-iteration lag raises clipfrac slightly as expected.
- **Caveat:** the first 2-GPU attempt hung in its first iteration (0 % GPU on both for 25 min, no stack available: ptrace denied);
  the identical rerun ran cleanly. `ppo_async_bench.py` now dumps all thread stacks every 4 min (faulthandler) to catch it if it recurs.
  Before using async in a long run: add the same faulthandler watchdog to `rl_train.py` and compare learning vs sync on a short run.

## Async learning A/B (2026-10-30, `reports/rl/q_async_ab.sh`)
HPO lr6e-6 recipe (P24c start, β 0.01, pool 9 + V33, 10M = 313 its) with `async_collect=true actor_device=cuda:1`
(learner GPU2, actor GPU0), learning-curve eval `hpo_eval.sh`, compared with the two sync replicates (`hpo_analysis.py`).
(The first launch died at iter 0 on a CUDA-graph capture-stream bug, fixed in f0690a3.)

| | iter_50 | iter_100 | iter_200 | final | curve | wall |
|---|---|---|---|---|---|---|
| async − lr6e-6 (sync) | −2.5±1.5 | −5.6±1.5 | +1.9±1.4 | **−4.6±1.2** | **−3.1±0.7** | 1.30 h vs 2.64 h |
| async − lr6e-6_r2 (sync) | −3.2±1.5 | −5.8±1.7 | +0.5±1.5 | **−3.0±1.3** | **−3.2±0.9** | 1.30 h vs 2.69 h |
| (noise: r2 − r1) | | | | −1.7±1.2 | +0.0±0.9 | |

- **Speed: 15.0 s/iter vs 30.4–30.9 (2.0×; the sync HPO runs shared the GPU with the previous trial's eval).**
- **Learning: degraded** — below both sync replicates by ~3 points (curve −3.1/−3.2, > 3 SE; final −4.6/−3.0), worst vs p24c
  (−10.3/−6.1) and V32.1 (−6.0/−5.2). Per update the async run is roughly back at the base lr-3e-6 level (final +1.6 vs base).
- Per wall-clock it is still ahead (async final at 1.3 h ≈ +28.7 vs sync ≈ +25–26 at the same wall time, it≈155), but the
  per-sample loss matters for long runs. Likely cause: the 1-iteration lag makes the clipped ratio π/π_behaviour start away
  from 1 (clipfrac ↑), i.e. fewer effective updates. Candidate fix: **decoupled PPO** (Hilton et al. 2021) — clip against the
  proximal policy (weights at update start) and correct the lag with a truncated importance weight π_prox/π_behaviour.
