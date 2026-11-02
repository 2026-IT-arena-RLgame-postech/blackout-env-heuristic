#!/bin/bash
# Post-patch RL (2026-11-01): the PPO league recipe (reports/rl/ppo_league_b.sh) from the official-information student D7
# (mask mode 3: enemy classes + storage amounts tracked in the ONNX state, floor/carried amounts hidden), sim with the
# official v1.0.2 collision.  Pool = post-patch heuristics (B3 x2, C0, V34c, V34b) + distilled students + gated self-play.
# Usage on poscat_gpu2 (never GPU 3):  INIT=~/bpruns/d7_final.pt GPU=2 MAXDEC=20000000 reports/bp/rl/ppo_bp1.sh
# Optional early teacher distillation (the heuristic alias is stepped on the learner's own states; CE coef decays linearly to 0 over teach_iters):
#   EXTRA="teach=B5 teach_coef=0.5 teach_iters=100"
set -e
cd ${REPO:-$HOME/blackout-env-bp}
FZ=$HOME/bpruns
export OPPS="${OPPS:-B3;B3;C0;V34c;V34b;net:$FZ/d3_final.pt;net:$FZ/d4_final.pt}"
export EVAL_OPPS="${EVAL_OPPS:-B3,C0,V34c,net:$FZ/d3_final.pt}"
export W=${W:-"1,1,1,1,0.5,1,0.5"}
export OUT=${OUT:-$HOME/bpruns/rl_bp1}
export EXTRA="beta=${BETA:-0.015} lr=${LR:-6e-6} init=${INIT:?set INIT} value=${VALUE:-$HOME/blackout-env-abl/checkpoints/value/s2v_f3xd2/iter2.pt} critic_warmup=${CWARM:-10} $EXTRA"
export REPO=$PWD
exec reports/rl/ppo_league_b.sh
