#!/bin/bash
# P20 256-seed gate (docs §47) on wsl_gpu: seeds 1100064-255 (64 + new 192) for the candidate AND the base P17, 8 opponents.
# Waits for the 64-seed eval (reports/p20_p19v5) and pilot B; then the report on all 256 seeds.
R=$HOME/blackout-env-p18b; OUT=$R/reports/p20_gate256; C=$HOME/p14frozen/p20.pt
until [ "$(grep -c "EVAL DONE" $R/reports/p20_p19v5/stdout.log)" -ge 2 ]; do sleep 30; done
while kill -0 $(cat $HOME/p14runs/p20_b/run.pid) 2>/dev/null; do sleep 30; done
cd $R; mkdir -p $OUT
export PYTHONPATH=$R NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
export BLACKOUT_NETS=p20=$C,p17=$HOME/p14frozen/p17.pt,p14b=$HOME/p14frozen/p14b.pt
V30="strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
V29="strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve"
V28="strategic_v28:reserve_stage=0.5,trade_hold=advance"
[ -e $OUT/matches.jsonl ] || sed "s/\"ours\": \"p19v5\"/\"ours\": \"p20\"/; s/\"opponent\": \"p19v5\"/\"opponent\": \"p20\"/" $R/reports/p20_p19v5/matches.jsonl > $OUT/matches.jsonl
OPP="${OPPS:-$V28;p17;$V30;strategic_v25;p14b;strategic_v23;$V29;strategic_v27:reserve_stage=1.5}"
for o in $(echo "$OPP" | tr ";" " "); do
  for us in p20 p17; do
    .venv/bin/python examples/v27_eval.py play --seeds 192 --seed0 1100064 --cap-s 425 --stop-exhausted --device cuda --chunk 16 \
      --workers ${EW:-11} --out $OUT --ours $us --opponent "$o"
  done
  .venv/bin/python examples/v27_eval.py report --base p17 --out $OUT > $OUT/report.txt
done
echo "GATE DONE" >> $OUT/stdout.log
