#!/bin/bash
# V32 (docs §58) Mac smoke: variants (OURS) vs OPP, SEEDS x 2 sides from 5200000.  Usage: mac_smoke.sh OUT SEEDS WORKERS
cd "$(dirname "$0")/../../.."
PY=${PY:-/Users/mac/project/26rl/blackout-env/.venv/bin/python}
export PYTHONPATH=. NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P="reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
V30="strategic_v30:$P"; V31="$V30,threat_radius=0.13,camp_engage_radius=6.5"
H4="$V30,hunter_quota=4"; A="strategic_v32:$P,threat_radius=0.13,camp_engage_radius=6.5,hunter_quota=4"
OURS=${OURS:-"$H4;$A;$A,exit_guards=1;$A,exit_guards=1,guard_min_hunters=3;$A,sweep=true;$A,exit_guards=1,sweep=true"}
nice -n 15 $PY examples/v27_eval.py play --seeds $2 --seed0 5200000 --cap-s 425 --stop-exhausted --chunk 8 --workers $3 \
  --out $1 --ours "$OURS" --opponent "${OPP:-$V31;$H4}" > $1.log 2>&1
$PY examples/v27_eval.py report --out $1 --base "$H4" > /dev/null
echo DONE >> $1.log
