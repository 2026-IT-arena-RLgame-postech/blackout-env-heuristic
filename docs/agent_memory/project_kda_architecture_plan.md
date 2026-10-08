---
name: project-kda-architecture-plan
description: "Planned KDA (Kimi Delta Attention) architecture for MyModel -- deferred, full design saved in-repo"
metadata: 
  node_type: memory
  type: project
  originSessionId: ab6dc1fa-eaa7-43d4-844d-16059b86488c
  modified: 2026-09-14T01:41:42.058Z
---

Full design doc saved at `blackout-env/blackout_env/model/kda_architecture_plan.md` on main (moved to `docs/design/kda_architecture_plan.md` on the run11-80k branch, 2026-09-19): a
[spatial encoder attention -> temporal KDA -> spatial decoder attention] hybrid the user
designed for MyModel, meant to add cross-timestep memory (the model is currently fully
stateless per env step). Explicitly deferred (2026-09-14) in favor of two smaller fixes first:
adding a per-unit identity embedding (`vec_slot_emb`, done) and fixing offline_pretrain.py's
n_step/gamma annealing to reset+re-anneal repeatedly like BBF instead of once per run (done).

**Why:** KDA is a much bigger change (new recurrent subsystem, replay-buffer windowed BPTT,
another checkpoint-breaking architecture bump) -- the user wanted the two contained fixes
landed and validated first.

**How to apply:** Before starting KDA work, read
the doc (path above, per branch) in full -- it has the exact
pipeline (encoder -> chunk-parallel KDA -> decoder), the train-vs-rollout compute-mode
table, and a section of BlackOut-specific gotchas already identified: no Triton on this Mac
(mps device, `triton` not installed) so KDA needs a pure-PyTorch chunk-parallel delta rule;
MyModel/MyPolicy/qmix_trainer are all stateless per-step today so rollout needs to carry KDA
state `S_t` across an episode (= one absorption interval, see
[[project_heuristic_variation_cadence]]) and training needs windowed BPTT (can likely reuse
the existing SPR k-step window sampling in `SequentialReplayBuffer`); watch that spatial
compression into h_t doesn't re-collapse the per-unit identity `vec_slot_emb` just fixed.

2026-09-14 refinement (also in the doc): encoder/decoder are both just the *existing*
MyModel full-attention trunk (47 tokens) run twice, with 5 learnable CLS tokens appended on
the encoder pass (-> h_t) and the KDA-refined M_t swapped in for those same 5 slots on the
decoder pass; Q is read directly from the 5 decoder output slots instead of from
`unit_out`'s 10 unit rows. Left as an explicit open question in the doc: whether those 5
slots are meant as one dedicated persistent memory per own-team unit (which would replace
`vec_slot_emb` outright) or a generic 5-way scene summary -- confirm with the user before
implementing the Q read-out.

**2026-09-30: decisions (user) and status.**

The user decided:
1. Goal: distill ANY heuristic, stateful ones included, as teacher.
2. Build opponent-conditional teachers.
3. Memory resets per match (not per absorption).
4. Extra inference latency is OK.
5. RL fine-tuning is unlikely for now.

Why memory is needed:
- 18–32% of the moves of V4, V17, V19 and the new strategies depend on history: a fresh instance given the same obs moves differently.
- V21-teach scores 0%.
- The history-dependence probe is in the session scratchpad only.

Status:
- Built a pure-PyTorch KDA layer (`model/modules/kda.py`) and a design doc (`docs/kda_temporal_design_ko.md`).
- In progress: option b2, a per-slot KDA between the trunk and the Q head, zero-init residual, warm-startable. Alongside it: `act:<heuristic>` teachers and sequence windows in the dataset.

**2026-09-30 results.**
- **P2 opponent-id probe** (reports/opponent_id_probe/): memory identifies the opponent family quickly.
  - The stateless probe stays at about 0.40 accuracy (11 classes). A GRU reaches 0.51 / 0.62 / 0.67 at 5 / 10 / 20 s; KDA and b2 are within noise of the GRU.
  - V17, V18, V19 and the checkpoints have recall 0.72–0.84 by 10 s.
- **Any-angle heuristic confirmed:** strategic_v21_anyangle − v21 = +32.9 ± 6.5 vs V19 and +25.9 ± 6.5 vs V17 (24 seeds). The path itself is about optimal (0.99–1.00× the visibility-graph optimum; 8-way following is 1.03×). What remains 8-way is target selection, which uses octile distances.
- **STAND ablation:** D5 − D5n is about −20 against V19-family opponents. STAND hurts even though it removes flicker, so D6 should start from D5n.
- **KDA fix:** the MPS doubling solve was unstable on repeated keys; it was fixed by a blocked solve (052d88d). Temporal runs made before the fix were exposed to it.
