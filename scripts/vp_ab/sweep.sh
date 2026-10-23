#!/bin/bash
# 2x2 {base, +dpos} x {mean-pool, attn-pool} value-net A/B on WSL (reports/value/vel_pool_ab.md).
# usage: scripts/vp_ab/sweep.sh [SEEDS...]   (default seeds 0 1 2); resumable via runs/sweep.log
# "done" is logged only when the run exited 0, printed its final "done:" line and left latest.pt.
cd ~/blackout-env-vp
export PYTHONPATH=.
PY=~/blackout-env/.venv/bin/python
run() { name=$1; shift
  grep -q "^$name done" runs/sweep.log 2>/dev/null && return; rm -rf runs/$name
  if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" != 0 ]; then echo "GPU busy before $name" >> runs/sweep.log; sleep 600; fi
  $PY examples/value_train.py train --cache datasets/value_s1_r64 --out-dir runs/$name --init checkpoints/distill/frozen/p10_v22.pt \
    --trunk-lr-scale 0.3 --steps 16000 --warmup 800 --lr 3e-4 --batch-size 256 --eval-every 4000 --eval-rows 20000 \
    --device cuda --amp fp16 --compile --final-head reg --final-beta 10 "$@" > runs/$name.log 2>&1
  rc=$?; sync
  if [ $rc = 0 ] && [ -s runs/$name/latest.pt ] && grep -q '^done:' runs/$name.log; then
    echo "$name done $(date)" >> runs/sweep.log
  else
    echo "$name FAILED rc=$rc $(date)" >> runs/sweep.log
  fi
  sync
}
for s in ${@:-0 1 2}; do
  run base_s$s --seed $s
  run vel_s$s --seed $s --vel
  run attn_s$s --seed $s --pool attn
  run velattn_s$s --seed $s --vel --pool attn
done
echo "SWEEP DONE $*" >> runs/sweep.log
sync
