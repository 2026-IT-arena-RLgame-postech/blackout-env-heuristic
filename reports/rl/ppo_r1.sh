#!/bin/bash
# PPO run r1 (docs/rl_finetune_plan_ko.md R1) vs GRPO r3b / r3c: same init (P24c), opponent pool + weights,
# PFSP hard, KL anchor, eval set (p24c, p22 in pool; p5a, p16 held out) and 10 h wall clock.
# PPO-specific: on-policy trunk games (games x segment decisions per iteration), separate critic warm-started
# from the S2 value net (blackout_env/rl/ppo.py), GAE(gamma 1, lambda 0.95), critic-only warm-up iterations.
# Budget: MAXDEC = r3b's total decisions (equal experience; the perf merge makes wall clock unequal).
# ~33k decisions/iter vs GRPO 166k: eval every 150 iters (~same decisions between evals as GRPO every 30).
# Usage on poscat_gpu2: MAXDEC=<n>  GPU=<0|1|2> reports/rl/ppo_r1.sh      (never GPU 3)
set -e
cd ~/blackout-env
GPU=${GPU:?set GPU}; [ "$GPU" != 3 ] || { echo "GPU 3 is off limits"; exit 1; }
UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$GPU '$1==g{print $2}')
busy=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u=$UUID '$1==u')
[ -z "$busy" ] || { echo "GPU $GPU busy: $busy"; exit 1; }
nvidia-smi -i $GPU | grep -q "No running processes found" || { nvidia-smi -i $GPU; echo "GPU $GPU has processes (incl. graphics)"; exit 1; }
F=checkpoints/distill/frozen
OUT=${OUT:-checkpoints/rl/ppo_r1}
mkdir -p $OUT
OPPS="net:$F/p24c_v321.pt;net:$F/p22_v31.pt;net:$F/p21e_v30.pt;net:$F/p20_v30.pt;net:$F/p17_v28.pt;net:$F/p14_v25.pt;net:$F/d6_v42.pt;V32.1;mixture"
W="0.5,1,1,1,1,1,1,1,1"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
CUDA_VISIBLE_DEVICES=$GPU setsid nohup nice -n 30 env PYTHONPATH=. .venv/bin/python examples/rl_train.py \
    --algo ppo --out $OUT --hours ${HOURS:-24} --max-decisions ${MAXDEC:?set MAXDEC} --eval-every ${EVAL_EVERY:-150} --eval-seeds ${EVAL_SEEDS:-16} \
    --set "opponents=$OPPS" "opp_weights=$W" pfsp=hard games=${GAMES:-256} segment=${SEGMENT:-128} ${EXTRA} \
    > $OUT/run.log 2>&1 &
echo "launched pid $! on GPU $GPU -> $OUT"
