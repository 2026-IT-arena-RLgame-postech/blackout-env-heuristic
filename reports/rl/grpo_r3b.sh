#!/bin/bash
# GRPO run r3b (after r3a, reports/rl/grpo_r3a_eval.md): r3a only gained vs its own init P24c (+9..13)
# and nothing vs P22, with R_group_std ~0.02.  Changes:
#   * pool: P24c down-weighted (0.5), + D6 and two heuristics (V32.1 = the distillation teacher, mixture); P5a / P16 held out
#   * PFSP "hard" (play more against what beats us)
#   * sample window 25 -> 75 decisions (1 s -> 3 s) for more within-group spread
#   * eval: p24c, p22 (in pool) + p5a, p16 (held out)
#   * from the GRPO paper check (Shao et al. 2024, G = 64): K 8 -> 16 at 48 roots (same 768 branches),
#     and drop near-flat groups (std < 0.005; logged as kept_frac)
# Usage on poscat_gpu2:  GPU=<0|1|2> reports/rl/grpo_r3b.sh      (never GPU 3)
#   r3c (same + value-net retraining on finished trunk games, 10 % replay):
#       GPU=<n> OUT=checkpoints/rl/grpo_r3c EXTRA="value_train=true" reports/rl/grpo_r3b.sh
#   r3d (value-free: branches to game end): GPU=<n> OUT=checkpoints/rl/grpo_r3d NGROUPS=16 K=16 EXTRA="leaf_steps=11000" MAXDEC=33327244 EVAL_EVERY=10 reports/rl/grpo_r3b.sh
set -e
cd ~/blackout-env
GPU=${GPU:?set GPU}; [ "$GPU" != 3 ] || { echo "GPU 3 is off limits"; exit 1; }
UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$GPU '$1==g{print $2}')
busy=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u=$UUID '$1==u')
[ -z "$busy" ] || { echo "GPU $GPU busy: $busy"; exit 1; }
nvidia-smi -i $GPU | grep -q "No running processes found" || { nvidia-smi -i $GPU; echo "GPU $GPU has processes (incl. graphics)"; exit 1; }
F=checkpoints/distill/frozen
OUT=${OUT:-checkpoints/rl/grpo_r3b}
mkdir -p $OUT
OPPS="net:$F/p24c_v321.pt;net:$F/p22_v31.pt;net:$F/p21e_v30.pt;net:$F/p20_v30.pt;net:$F/p17_v28.pt;net:$F/p14_v25.pt;net:$F/d6_v42.pt;V32.1;mixture"
W="0.5,1,1,1,1,1,1,1,1"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
# r3d OOM-ed at iter 40 (fragmentation over many CUDA-graph / branch batch sizes): growable segments
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
CUDA_VISIBLE_DEVICES=$GPU setsid nohup nice -n 30 env PYTHONPATH=. .venv/bin/python examples/rl_train.py \
    --algo grpo --out $OUT --hours ${HOURS:-10} --max-decisions ${MAXDEC:-inf} --eval-every ${EVAL_EVERY:-30} --eval-seeds ${EVAL_SEEDS:-16} \
    --set "opponents=$OPPS" "opp_weights=$W" pfsp=hard sample_steps=75 groups=${NGROUPS:-48} k=${K:-16} drop_flat=${DROP_FLAT:-0.005} ${EXTRA} \
    > $OUT/run.log 2>&1 &
echo "launched pid $! on GPU $GPU -> $OUT"
