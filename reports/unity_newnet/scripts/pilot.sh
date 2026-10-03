#!/usr/bin/env bash
# GPU vs CPU for the big nets: P11 vs V23, 24 matches, 12 Unity workers, each way (docs §22 "Unity 확인").
cd ~/blackout-env-newnet
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1
PY=~/blackout-env/.venv/bin/python
for mode in cpu gpu; do
  if [[ $mode == gpu ]]; then export BLACKOUT_CUDA_NETS=p11_v22.pt,p5a_v22.pt; else unset BLACKOUT_CUDA_NETS; fi
  t0=$(date +%s)
  $PY examples/distill_heldout_eval.py checkpoints/distill/frozen/p11_v22.pt --opponents strategic_v23 \
    --teacher act:strategic_v23 --no-teacher-baseline --seeds 12 --seed-base 7700000 --workers 12 \
    --build build/linux_server/BlackOut.x86_64 --out /tmp/newnet_pilot_$mode > /tmp/newnet_pilot_$mode.log 2>&1
  echo "$mode $(( $(date +%s) - t0 )) s" >> /tmp/newnet_pilot_times.txt
done
echo PILOT DONE >> /tmp/newnet_pilot_times.txt
