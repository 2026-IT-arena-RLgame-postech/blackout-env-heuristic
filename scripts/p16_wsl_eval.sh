#!/bin/bash
# P16 simulator eval chain on wsl_gpu (docs/policy_v2_design_ko.md §35): waits for the P16 run to finish, freezes it,
# then plays examples/v27_eval.py (full matches, stop when exhausted; held-out seeds 1100000-127 x both sides = 256
# matches per pairing): ours p16 / p14b (paired base) / V27 teacher (reference) vs V25, V27(1.5), P14b, V23, P11, V19.
# Records first Hunter decision, second race, reserve distance at decision 99.
# Usage: scripts/p16_wsl_eval.sh <run-dir> [workers]   (detached; pid in ~/p16tmp/eval.pid)
R=$HOME/blackout-env-p16
B=$1
W=${2:-11}
OUT=$R/reports/p16_sim
mkdir -p $OUT $HOME/p16tmp
cd $R
export PYTHONPATH=$R NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 BLACKOUT_NETS=p16=$HOME/p14frozen/p16.pt,p14b=$HOME/p14frozen/p14b.pt,p11=$HOME/p14assets/checkpoints/distill/frozen/p11_v22.pt
V27='strategic_v27:reserve_stage=1.5'
setsid nohup bash -c "
  while kill -0 \$(cat $B/run.pid) 2>/dev/null; do sleep 30; done
  grep -q 'done:' $B/stdout.log && [ ! -e $HOME/p14frozen/p16.pt ] && cp $B/latest.pt $HOME/p14frozen/p16.pt
  [ -e $HOME/p14frozen/p16.pt ] || { echo 'NO P16 CHECKPOINT' >> $OUT/stdout.log; exit 1; }
  .venv/bin/python examples/v27_eval.py play --ours 'p16;p14b;$V27' \
      --opponent 'strategic_v25;$V27;p14b;strategic_v23;p11;strategic_v19' --seeds 128 --seed0 1100000 \
      --cap-s 425 --stop-exhausted --device cuda --chunk 32 --workers $W --out $OUT >> $OUT/stdout.log 2>&1
  .venv/bin/python examples/v27_eval.py report --base p14b --out $OUT >> $OUT/stdout.log 2>&1
  echo 'P16 EVAL DONE' >> $OUT/stdout.log" > /dev/null 2>&1 < /dev/null &
echo $! > $HOME/p16tmp/eval.pid
echo "eval chain pid $(cat $HOME/p16tmp/eval.pid)"
