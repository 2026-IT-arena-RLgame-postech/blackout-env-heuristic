#!/bin/bash
# L1 chain on gpu2 GPU2: wait until no other user's process at two checks 10 min apart -> train (ppo_l1.sh) -> OOD-26 (CPU, bg)
# + GPU evals (abl_eval TAG l1 CTRL F2; eval_fast tag l1 vs F3, V33 + hunt OOD into reports/rl/angle_probe). Log reports/rl/q_l1.log.
cd ~/blackout-env-abl; GPU=2; L=reports/rl/q_l1.log; F=checkpoints/distill/frozen
foreign() { local u=$(nvidia-smi -i $GPU --query-gpu=uuid --format=csv,noheader) c=0
  for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F", " -v u="$u" '$1==u{print $2}'); do
    [ "$(ps -o user= -p $p | tr -d ' ')" = "$USER" ] || c=$((c+1)); done; echo $c; }
waitfree() { while true; do [ "$(foreign)" = 0 ] || { sleep 600; continue; }; sleep 600; [ "$(foreign)" = 0 ] && break; done; }
waitfree; echo "$(date) launching L1" >> $L
ALLOW_OWN=1 REPO=~/blackout-env-abl GPU=$GPU bash reports/rl/ppo_l1.sh >> $L 2>&1
pid=$(sed -n 's/launched pid \([0-9]*\).*/\1/p' $L | tail -1)
while kill -0 $pid 2>/dev/null; do sleep 120; done
[ -s checkpoints/rl/l1_league/final.pt ] || { echo "$(date) L1 died without final.pt" >> $L; exit 1; }
echo "$(date) L1 trained" >> $L
D=reports/rl/abl_ood_l1; mkdir -p $D
(PYTHONPATH=. NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BLACKOUT_NETS=p21e=$F/p21e_v30.pt,p20=$F/p20_v30.pt,p17=$F/p17_v28.pt,p14b=$F/p14_v25.pt nice -n 15 .venv/bin/python examples/v27_eval.py play --seeds 32 --seed0 2600000 --cap-s 425 --stop-exhausted --device cpu --chunk 16 --workers 10 --out $D --ours "net:checkpoints/rl/l1_league/final.pt" --opponent "ood26=v30_h4hunt,v30_h2,v30_camp,v30_stage3,v30_nostage,v30_c0,v30_hunt,v30_reckless,early_rush,fortress,raider,duelist,cargo_hunter,rush_all,v4,v19,v21,v26,random,idle,v31_eps10,v30_lag4,v30_freeze10,p20_t1,v31" > $D/ood26.log 2>&1; echo "$(date) L1 OOD done" >> $L) &
waitfree; echo "$(date) L1 eval start" >> $L
GPU=$GPU CAND=checkpoints/rl/l1_league/final.pt TAG=l1 CTRL=checkpoints/rl/f2_lr6/final.pt bash reports/rl/abl_eval.sh >> $L 2>&1
OOD=$(PYTHONPATH=. .venv/bin/python -c "
from blackout_env.train.distill.ood_suite import OOD26, REFERENCE, FOLLOWUP
d={n:x for n,x,_,_ in OOD26+REFERENCE+FOLLOWUP}
print(';'.join(d[n] for n in ['v30_h4hunt','v30_hunt','v30_reckless','early_rush','rush_all']))")
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 .venv/bin/python examples/eval_fast.py --cand checkpoints/rl/l1_league/final.pt \
  --opps "net:checkpoints/rl/f3_lr8/final.pt;V33;$OOD" --lo 970000 --hi 970256 --chunk 128 --out reports/rl/angle_probe --tag l1 --device cuda:0 --heur-workers 8 > reports/rl/angle_probe/l1.log 2>&1
echo "$(date) L1 eval done" >> $L
wait
