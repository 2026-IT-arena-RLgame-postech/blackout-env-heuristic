#!/bin/bash
# §25 Unity: ft_f3 / ft_base vs V23 and V19, seeds 1200000-1200199 x both sides (= §19 P11 rows), Linux server build.
cd ~/blackout-env-p11m; export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=2
exec ~/blackout-env/.venv/bin/python examples/distill_heldout_eval.py runs/models/ft_f3.pt runs/models/ft_base.pt \
  --teacher act:strategic_v23 --opponents strategic_v23 strategic_v19 --seeds 200 --seed-base 1200000 \
  --no-teacher-baseline --build build/linux_server/BlackOut.x86_64 --workers ${WORKERS:-10} --out runs/unity
