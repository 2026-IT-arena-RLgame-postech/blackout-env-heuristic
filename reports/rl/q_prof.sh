#!/bin/bash
# PPO profile (reports/rl/ppo_perf2.md) on gpu2 GPU2 once no other user's process at two checks 10 min apart.
# Current recipe: L1 pool, fp16 updates + compile, games 256 x segment 128, heur_workers 6; init/critic = L1.
cd ~/blackout-env-abl; GPU=2; L=reports/rl/prof/q_prof.log; mkdir -p reports/rl/prof
foreign() { local u=$(nvidia-smi -i $GPU --query-gpu=uuid --format=csv,noheader) c=0
  for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F", " -v u="$u" '$1==u{print $2}'); do
    [ "$(ps -o user= -p $p | tr -d ' ')" = "$USER" ] || c=$((c+1)); done; echo $c; }
while true; do [ "$(foreign)" = 0 ] || { sleep 600; continue; }; sleep 600; [ "$(foreign)" = 0 ] && break; done
F=checkpoints/distill/frozen; R=checkpoints/rl
OPPS="net:$F/p24c_v321.pt;net:$F/p22_v31.pt;net:$F/p21e_v30.pt;net:$F/p20_v30.pt;net:$F/p17_v28.pt;net:$F/p14_v25.pt;net:$F/d6_v42.pt;V32.1;mixture;V33;net:$R/f2_lr6/final.pt;net:$R/f3_lr8/final.pt"
SET="update_fp16=true critic_fp16=true update_compile=true critic_head=reg heur_workers=6 update_micro=512 critic_read_batch=1024 critic_warmup=0 beta=0.01 lr=6e-6 init=$R/l1_league/final.pt critic_init=$R/l1_league/final_critic.pt"
echo "$(date) profile start" >> $L
for mode in nowrap wrap; do
  X=""; [ $mode = nowrap ] && X="--no-wrap"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 .venv/bin/python examples/ppo_profile.py --iters 5 $X \
    --opp-specs "$OPPS" --weights "0.5,1,1,1,1,1,1,1,1,1,1,1" --set $SET > reports/rl/prof/$mode.jsonl 2> reports/rl/prof/$mode.err
  echo "$(date) $mode done rc=$?" >> $L
done
uptime >> $L
