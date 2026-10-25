#!/bin/bash
# PPO ablations on top of the league-B recipe (reports/rl/ppo_league_b.sh), fixed 10M-decision budget
# (PPO plateaus at ~5-10M decisions; compare lambda arms with league_b/iter_300.pt ~ 9.8M decisions,
# KL arms (20M) with league_b/iter_600.pt; league_b2 = second seed of B = run-to-run noise floor).
#   lam99   : gae_lambda 0.99,  segment 256, games 128 (same 32k rows/it; longer segment so lambda is not cut by the bootstrap)
#   lam995  : gae_lambda 0.995, segment 512, games 64
#   refreset: KL anchor pi_ref <- learner at each snapshot (league_ref_reset)
#   beta01  : KL coefficient 0.02 -> 0.01
# Usage: ARM=<arm> [REPO=... GPU=...] reports/rl/ppo_ablate.sh   (GPU check is in ppo_league_b.sh; never GPU 3)
set -e
ARM=${ARM:?set ARM}
case $ARM in
  lam99)    export SEGMENT=256 GAMES=128 EXTRA="gae_lambda=0.99";;
  lam995)   export SEGMENT=512 GAMES=64  EXTRA="gae_lambda=0.995";;
  refreset) export EXTRA="league_ref_reset=true" MAXDEC=${MAXDEC:-20000000};;   # KL arms: 20M (snapshots only from it 150; plateau at ~it 300)
  beta01)   export EXTRA="beta=0.01" MAXDEC=${MAXDEC:-20000000};;
  *) echo "bad ARM $ARM"; exit 1;;
esac
export OUT=${OUT:-checkpoints/rl/abl_$ARM} MAXDEC=${MAXDEC:-10000000}
exec bash "$(dirname "$0")/ppo_league_b.sh"
