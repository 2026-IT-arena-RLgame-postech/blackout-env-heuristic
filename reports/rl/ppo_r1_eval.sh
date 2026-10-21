#!/bin/bash
# Waits for PPO r1 to finish, then runs its 256-seed paired eval (seeds 970000-970255, both sides, 64-seed chunks)
# vs the same opponents as reports/rl/grpo_r3bc_eval (p24c, p22, V32.1, p5a, p16) -> same dir, tag ppo_final.
# GPUs 0 and 2 only, each re-checked (no other processes) right before use; 3 jobs per GPU.
# Usage on poscat_gpu2:  setsid nohup reports/rl/ppo_r1_eval.sh > reports/rl/grpo_r3bc_eval/ppo_driver.log 2>&1 &
cd ~/blackout-env
D=reports/rl/grpo_r3bc_eval; F=checkpoints/distill/frozen; CK=checkpoints/rl/ppo_r1/final.pt
while pgrep -f "out checkpoints/rl/ppo_r1" >/dev/null || [ ! -s $CK ]; do
  pgrep -f "out checkpoints/rl/ppo_r1" >/dev/null || { echo "$(date) PPO gone but no $CK"; exit 1; }
  sleep 120
done
echo "$(date) PPO finished"
GPUS=""
for g in 0 2; do
  if nvidia-smi -i $g | grep -q "No running processes found"; then GPUS="$GPUS $g"; else echo "GPU $g busy, skipped"; nvidia-smi -i $g; fi
done
[ -n "$GPUS" ] || { echo "no free GPU"; exit 1; }
echo "$(date) using GPUs:$GPUS"
jobs=()
for o in "p24c_v321 net:$F/p24c_v321.pt" "p22_v31 net:$F/p22_v31.pt" "p5a_v22 net:$F/p5a_v22.pt" "p16_v27 net:$F/p16_v27.pt" "v32_1 V32.1"; do
  for lo in 970000 970064 970128 970192; do jobs+=("$o $lo $((lo + 64))"); done
done
i=0
for j in "${jobs[@]}"; do echo "$j"; done | while read ot os lo hi; do
  set -- $GPUS; n=$#; g=$(echo $GPUS | cut -d' ' -f$(( i % n + 1 ))); i=$((i + 1))
  echo "$g $ot $os $lo $hi"
done | xargs -P $(( $(echo $GPUS | wc -w) * 3 )) -L 1 bash -c '
  g=$0; ot=$1; os=$2; lo=$3; hi=$4; out='$D'/ppo_final__${ot}__${lo}.json
  [ -s $out ] && exit 0
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 30 env PYTHONPATH=. .venv/bin/python examples/eval_grpo_ckpt.py \
    --cand '$CK' --opp $os --lo $lo --hi $hi --out $out --device cuda:0 > '$D'/logs/ppo_final__${ot}__${lo}.log 2>&1
  echo "$(date) done $ot $lo rc=$?"'
echo "$(date) ALL DONE $(ls $D/ppo_final__*.json | wc -l)/20"
