#!/bin/bash
# PPO hyper-parameter sweep (2026-10-27): F1 recipe (beta 0.01, league B, pool 9 + V33) from the distilled P24c, 10M decisions,
# checkpoints every 50 its (iter_50.pt ...) so reports/rl/hpo_eval.sh can measure the learning curve (speed) and the end point.
# One change per trial vs the "base" trial; HP = extra key=value settings (parse_set: last wins).
# Usage: TRIAL=<name> HP="lr=6e-6" [REPO=.. GPU=..] reports/rl/ppo_hpo.sh
set -e
F=checkpoints/distill/frozen
export OPPS="net:$F/p24c_v321.pt;net:$F/p22_v31.pt;net:$F/p21e_v30.pt;net:$F/p20_v30.pt;net:$F/p17_v28.pt;net:$F/p14_v25.pt;net:$F/d6_v42.pt;V32.1;mixture;V33"
export W="0.5,1,1,1,1,1,1,1,1,1"
export EXTRA="beta=0.01 $HP"
export OUT=checkpoints/rl/hpo_${TRIAL:?} MAXDEC=${MAXDEC:-10000000} EVAL_EVERY=${EVAL_EVERY:-50} EVAL_SEEDS=${EVAL_SEEDS:-4}
exec bash "$(dirname "$0")/ppo_league_b.sh"
