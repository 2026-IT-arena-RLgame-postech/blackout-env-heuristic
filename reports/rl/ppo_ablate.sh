#!/bin/bash
# PPO ablations on top of the league-B recipe (reports/rl/ppo_league_b.sh), fixed 10M-decision budget
# (PPO plateaus at ~5-10M decisions; compare lambda arms with league_b/iter_300.pt ~ 9.8M decisions,
# KL arms (20M) with league_b/iter_600.pt; league_b2 = second seed of B = run-to-run noise floor).
#   lam99   : gae_lambda 0.99,  segment 256, games 128 (same 32k rows/it; longer segment so lambda is not cut by the bootstrap)
#   lam995  : gae_lambda 0.995, segment 512, games 64
#   refreset: KL anchor pi_ref <- learner at each snapshot (league_ref_reset)
#   beta01  : KL coefficient 0.02 -> 0.01
#   offset  : angle offset sampled N(mu, sigma), learnable sigma, in the PPO ratio + KL (offset_rl)
#   fine    : hierarchical 32 x 32 direction head (fine_bins=32: 1024 directions, ~0.35 deg sub-bins; blackout_env/rl/finehead.py)
# BETA=<x> (any arm) appends "beta=<x>" to EXTRA, e.g. ARM=fine BETA=0.01 (the E4 winner) for the combined arm.
# Usage: ARM=<arm> [REPO=... GPU=...] reports/rl/ppo_ablate.sh   (GPU check is in ppo_league_b.sh; never GPU 3)
set -e
ARM=${ARM:?set ARM}
case $ARM in
  lam99)    export SEGMENT=256 GAMES=128 EXTRA="gae_lambda=0.99";;
  lam995)   export SEGMENT=512 GAMES=64  EXTRA="gae_lambda=0.995";;
  refreset) export EXTRA="league_ref_reset=true" MAXDEC=${MAXDEC:-20000000};;   # KL arms: 20M (snapshots only from it 150; plateau at ~it 300)
  beta01)   export EXTRA="beta=0.01" MAXDEC=${MAXDEC:-20000000};;
  beta005)  export EXTRA="beta=0.005" MAXDEC=${MAXDEC:-20000000};;   # follow-ups of E4 (beta 0.01 won, +4.6±1.1 vs B it600)
  beta0)    export EXTRA="beta=0.0" MAXDEC=${MAXDEC:-20000000};;
  offset)   export EXTRA="offset_rl=true";;   # E5: Gaussian angle offset (eager update, no compile); CUDA smoke first
  fine)     export EXTRA="fine_bins=32 fine_lr=2e-5 fine_tau=1.5";;   # CUDA smokes: tau 0.5 -> kl_fine 0.08 / clipfrac 0.08 after 7 its (prior centred on the drifting angle head; lr irrelevant); tau 1.5 -> 0.0074 / 0.013, ent_fine 1.82   # eager update (no compile), like offset: CUDA smoke first
  *) echo "bad ARM $ARM"; exit 1;;
esac
[ -z "$BETA" ] || export EXTRA="$EXTRA beta=$BETA"
export OUT=${OUT:-checkpoints/rl/abl_$ARM} MAXDEC=${MAXDEC:-10000000} EVAL_SEEDS=${EVAL_SEEDS:-4}   # in-run evals are only a sanity check (16 seeds took ~25 min on gpu2); arms get a separate 256-seed eval
exec bash "$(dirname "$0")/ppo_league_b.sh"
