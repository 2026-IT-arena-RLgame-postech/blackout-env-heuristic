#!/bin/bash
# OOD1 (docs §56) best-response / cycle check: the V30 Hunter variants as CANDIDATES vs the gate pool + style policies,
# and the missing net-vs-net cells of the win-margin matrix.  32 seeds x 2 sides from 2600000 (= the screen's seeds).
# Usage: br.sh <part: br|mx> ; env EW (workers).  Out: reports/ood1/br.
R=${R:-$HOME/blackout-env-ood1}; OUT=$R/reports/ood1/br; F=$R/checkpoints/distill/frozen
cd $R; mkdir -p $OUT
export PYTHONPATH=$R NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
export BLACKOUT_NETS=p21e=$F/p21e_v30.pt,p20=$F/p20_v30.pt,p17=$F/p17_v28.pt,p14b=$F/p14_v25.pt
V30="strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
V31="$V30,threat_radius=0.13,camp_engage_radius=6.5"
H4H="$V30,hunter_quota=4,hunter_mode=hunt"; HU="$V30,hunter_mode=hunt"; H4="$V30,hunter_quota=4"
POOL="$V31;$V30;strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve;strategic_v28:reserve_stage=0.5,trade_hold=advance"
POOL="$POOL;strategic_v27:reserve_stage=1.5;strategic_v25;strategic_v23;p14b;p17;p20;p21e"
STYLE="strategic_early_rush;strategic_fortress;strategic_raider;strategic_duelist;strategic_cargo_hunter;strategic_v19"
if [ "$1" = br ]; then OURS="$H4H;$HU;$H4"; OPP="$POOL;$STYLE;$H4H;$HU;$H4"
else OURS="p21e;p17;strategic_v25"; OPP="p20;p17;p21e"; fi
nice -n 39 .venv/bin/python examples/v27_eval.py play --seeds ${SEEDS:-32} --seed0 2600000 --cap-s 425 --stop-exhausted \
  --device cpu --chunk 16 --workers ${EW:-8} --out $OUT --ours "$OURS" --opponent "$OPP"
echo "BR $1 DONE" >> $OUT/stdout.log
