#!/bin/bash
# sync vs async_collect (1 GPU, 2 streams) vs async on 2 GPUs (learner GPU2, actor GPU0); launches only when GPU2 and GPU0
# have no other user's process at two checks 10 min apart. Out: reports/rl/prof/async_{sync,a1,a2}.jsonl
cd ~/blackout-env-abl; L=reports/rl/prof/q_async.log; mkdir -p reports/rl/prof
foreign() { local u=$(nvidia-smi -i $1 --query-gpu=uuid --format=csv,noheader) c=0
  for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F", " -v u="$u" '$1==u{print $2}'); do
    [ "$(ps -o user= -p $p | tr -d ' ')" = "$USER" ] || c=$((c+1)); done; echo $c; }
free2() { [ "$(foreign 2)" = 0 ] && [ "$(foreign 0)" = 0 ]; }
while true; do free2 || { sleep 600; continue; }; sleep 600; free2 && break; done
F=checkpoints/distill/frozen; R=checkpoints/rl
OPPS="net:$F/p24c_v321.pt;net:$F/p22_v31.pt;net:$F/p21e_v30.pt;net:$F/p20_v30.pt;net:$F/p17_v28.pt;net:$F/p14_v25.pt;net:$F/d6_v42.pt;V32.1;mixture;V33;net:$R/f2_lr6/final.pt;net:$R/f3_lr8/final.pt"
SET="update_fp16=true critic_fp16=true update_compile=true critic_head=reg heur_workers=6 update_micro=512 critic_read_batch=1024 critic_warmup=0 beta=0.01 lr=6e-6 init=$R/l1_league/final.pt critic_init=$R/l1_league/final_critic.pt graph_shared_pool=true"
run() { echo "$(date) $1 start" >> $L
  CUDA_VISIBLE_DEVICES=$2 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 .venv/bin/python examples/ppo_async_bench.py --iters 7 $3 \
    --opp-specs "$OPPS" --weights "0.5,1,1,1,1,1,1,1,1,1,1,1" --set $SET > reports/rl/prof/async_$1.jsonl 2> reports/rl/prof/async_$1.err
  echo "$(date) $1 rc=$?" >> $L; }
#run sync
#run a1
run a2 2,0 "--async --actor-device cuda:1"
