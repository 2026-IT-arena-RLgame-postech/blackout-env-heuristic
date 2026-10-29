#!/bin/bash
# F3 chain: launch (q_f3.sh, GPU-free checks) -> wait for final.pt -> OOD-26 on CPU (bg) + GPU evals (q_f3eval.sh).
cd ~/blackout-env-abl; L=reports/rl/q_f3.log; F=checkpoints/distill/frozen
bash scripts/q_f3.sh
pid=$(sed -n "s/launched pid \([0-9]*\).*/\1/p" $L | tail -1)
while kill -0 $pid 2>/dev/null; do sleep 120; done
[ -s checkpoints/rl/f3_lr8/final.pt ] || { echo "$(date) F3 died without final.pt" >> $L; exit 1; }
echo "$(date) F3 trained" >> $L
D=reports/rl/abl_ood_f3; mkdir -p $D
(PYTHONPATH=. NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BLACKOUT_NETS=p21e=$F/p21e_v30.pt,p20=$F/p20_v30.pt,p17=$F/p17_v28.pt,p14b=$F/p14_v25.pt nice -n 15 .venv/bin/python examples/v27_eval.py play --seeds 32 --seed0 2600000 --cap-s 425 --stop-exhausted --device cpu --chunk 16 --workers 10 --out $D --ours "net:checkpoints/rl/f3_lr8/final.pt" --opponent "ood26=v30_h4hunt,v30_h2,v30_camp,v30_stage3,v30_nostage,v30_c0,v30_hunt,v30_reckless,early_rush,fortress,raider,duelist,cargo_hunter,rush_all,v4,v19,v21,v26,random,idle,v31_eps10,v30_lag4,v30_freeze10,p20_t1,v31" > $D/ood26.log 2>&1; echo "$(date) F3 OOD done" >> $L) &
bash scripts/q_f3eval.sh
wait
