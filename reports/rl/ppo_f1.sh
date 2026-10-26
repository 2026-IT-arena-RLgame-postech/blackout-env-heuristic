#!/bin/bash
# F1: continue PPO from E4 (abl_beta01/final.pt, beta 0.01 winner) with the new V33 heuristic added to the opponent pool.
#   init = KL reference = E4 (pi_ref is a copy of init); critic warm-started from E4's final critic; beta 0.01; league-B recipe;
#   pool = the 9 B opponents + V33 (weight 1); 20M decisions. Combos fine/offset (+beta 0.01) lost, so plain beta 0.01.
# Usage: [REPO=... GPU=... OUT=...] reports/rl/ppo_f1.sh   (GPU check in ppo_league_b.sh; never GPU 3)
set -e
F=checkpoints/distill/frozen
E4=${E4:-checkpoints/rl/abl_beta01}
export OPPS="net:$F/p24c_v321.pt;net:$F/p22_v31.pt;net:$F/p21e_v30.pt;net:$F/p20_v30.pt;net:$F/p17_v28.pt;net:$F/p14_v25.pt;net:$F/d6_v42.pt;V32.1;mixture;V33"
export W="0.5,1,1,1,1,1,1,1,1,1"
export EXTRA="beta=0.01 init=$E4/final.pt critic_init=$E4/final_critic.pt critic_warmup=${CWARM:-3} $EXTRA"
export OUT=${OUT:-checkpoints/rl/f1_v33} MAXDEC=${MAXDEC:-20000000} EVAL_SEEDS=${EVAL_SEEDS:-4}
exec bash "$(dirname "$0")/ppo_league_b.sh"
