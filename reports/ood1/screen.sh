#!/bin/bash
# OOD1 (docs §56) screen on poscat_gpu2 CPU: candidates OURS (';'-separated) vs the OOD-26 suite (OPP, default ood26),
# SEEDS (32) x 2 sides from SEED0 (2600000), cap 425 s, --stop-exhausted.  nice 19 (sessions start at -20), 1 thread each.
# Usage: screen.sh <out dir under reports/> ; env OURS, OPP, SEEDS, SEED0, EW (workers, 12).  Detached by the caller (setsid nohup).
R=${R:-$HOME/blackout-env-ood1}; OUT=$R/reports/$1; F=$R/checkpoints/distill/frozen
cd $R; mkdir -p $OUT
export PYTHONPATH=$R NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
export BLACKOUT_NETS=p21e=$F/p21e_v30.pt,p20=$F/p20_v30.pt,p17=$F/p17_v28.pt,p14b=$F/p14_v25.pt
V30="strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
V31="$V30,threat_radius=0.13,camp_engage_radius=6.5"
nice -n 39 .venv/bin/python examples/v27_eval.py play --seeds ${SEEDS:-32} --seed0 ${SEED0:-2600000} --cap-s 425 --stop-exhausted \
  --device cpu --chunk ${CHUNK:-16} --workers ${EW:-12} --out $OUT --ours "${OURS:-p21e;p20;$V30;$V31}" --opponent "${OPP:-ood26}"
.venv/bin/python examples/ood_report.py --out $OUT --base p20 > /dev/null
echo "SCREEN DONE" >> $OUT/stdout.log
