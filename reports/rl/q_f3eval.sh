#!/bin/bash
# F3 evals on GPU2 once it has no other user process at two checks 10 min apart: abl_eval vs F1 (+p24c/p22/V32.1), then V33 + hunt OOD (tag f2, pairs with angle_probe/f1__*).
cd ~/blackout-env-abl; GPU=2; L=reports/rl/q_f3.log
foreign() { local u=$(nvidia-smi -i $GPU --query-gpu=uuid --format=csv,noheader) c=0
  for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F", " -v u="$u" "\$1==u{print \$2}"); do
    [ "$(ps -o user= -p $p | tr -d " ")" = "$USER" ] || c=$((c+1)); done; echo $c; }
while true; do [ "$(foreign)" = 0 ] || { sleep 600; continue; }; sleep 600; [ "$(foreign)" = 0 ] && break; done
echo "$(date) F3 eval start" >> $L
GPU=$GPU CAND=checkpoints/rl/f3_lr8/final.pt TAG=f3_lr8 CTRL=checkpoints/rl/f2_lr6/final.pt bash reports/rl/abl_eval.sh >> $L 2>&1
F=checkpoints/distill/frozen
OOD=$(PYTHONPATH=. .venv/bin/python -c "
from blackout_env.train.distill.ood_suite import OOD26, REFERENCE, FOLLOWUP
d={n:x for n,x,_,_ in OOD26+REFERENCE+FOLLOWUP}
print(\";\".join(d[n] for n in [\"v30_h4hunt\",\"v30_hunt\",\"v30_reckless\",\"early_rush\",\"rush_all\"]))")
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 .venv/bin/python examples/eval_fast.py --cand checkpoints/rl/f3_lr8/final.pt --opps "V33;$OOD" \
  --lo 970000 --hi 970256 --chunk 128 --out reports/rl/angle_probe --tag f3 --device cuda:0 --heur-workers 8 > reports/rl/angle_probe/f3.log 2>&1
echo "$(date) F3 eval done" >> $L
