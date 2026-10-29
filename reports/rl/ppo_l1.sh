#!/bin/bash
# L1 league: continue from F2 (champion) with F2 and F3 (non-transitive pair: F3 beats F2 H2H +25 but loses V33 -12) added
# to the opponent pool. init = KL ref = F2, critic F2, beta 0.01, lr 6e-6 (HPO default), pool = 9 B + V33 + F2 + F3, 20M.
# Usage: [REPO=... GPU=... OUT=... LR=...] reports/rl/ppo_l1.sh   (GPU check in ppo_league_b.sh; never GPU 3)
set -e
F=checkpoints/distill/frozen; R=checkpoints/rl
INIT=${INIT:-$R/f2_lr6}
export OPPS="net:$F/p24c_v321.pt;net:$F/p22_v31.pt;net:$F/p21e_v30.pt;net:$F/p20_v30.pt;net:$F/p17_v28.pt;net:$F/p14_v25.pt;net:$F/d6_v42.pt;V32.1;mixture;V33;net:$R/f2_lr6/final.pt;net:$R/f3_lr8/final.pt"
export W="0.5,1,1,1,1,1,1,1,1,1,1,1"
export EXTRA="beta=0.01 lr=${LR:-6e-6} init=$INIT/final.pt critic_init=$INIT/final_critic.pt critic_warmup=${CWARM:-3} $EXTRA"
export OUT=${OUT:-checkpoints/rl/l1_league} MAXDEC=${MAXDEC:-20000000} EVAL_SEEDS=${EVAL_SEEDS:-4}
exec bash "$(dirname "$0")/ppo_league_b.sh"
