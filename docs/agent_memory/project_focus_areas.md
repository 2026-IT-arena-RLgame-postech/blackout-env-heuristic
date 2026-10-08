---
name: project-focus-areas
description: "What matters most in this project for docs/cleanup/review — Unity obs changes, reward model, heuristics; model/training loop secondary"
metadata: 
  node_type: memory
  type: project
  originSessionId: d0a20a83-92a2-4523-8046-4908277166ac
  modified: 2026-09-19T04:33:28.542Z
---

When reviewing, documenting or cleaning the project (e.g. preparing the run11-80k branch for teammates, 2026-09-19), the user wants the main attention on three areas:
1. Unity-side observation changes (blackout repo: MapObsAgent / SemanticMapRenderer / bit-packed graphic, shared state vector; Python side blackout_env/env/my_obs_preprocessor.py)
2. The reward model (Python reward v2 in blackout_env/train/reward_v2.py, docs/reward_v2_design.md; Unity-side Psi/Phi shaping it replaced)
3. Heuristics (blackout_env/heuristics/, V1–V19, collection mixture)

Model architecture (MyModel, attention, IQN) and the training loop (qmix_trainer, offline_pretrain) are secondary.

**Why:** these three are what teammates build on; the network/trainer are more replaceable.
**How to apply:** prioritize depth, accuracy and explanation quality in those three areas; don't spend most effort on model/trainer internals. Related: [[project-kda-architecture-plan]].
