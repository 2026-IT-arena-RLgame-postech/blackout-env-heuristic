#!/bin/bash
# Waits for both PPO r2 arms (reg: GPU 0, hlg: GPU 2; run from ~/blackout-env-ppodist, checkpoints shared with the
# mirror) to finish, then runs the 256-seed paired eval (seeds 970000-970255, both sides, 64-seed chunks) of each final
# vs p24c, p22, V32.1, p5a, p16 -> reports/rl/grpo_r3bc_eval/ppo_r2_{reg,hlg}__<opp>__<lo>.json (same games as r1/r3b/r3c).
# GPUs 0 and 2 only, each re-checked (no other processes) right before use; 3 jobs per GPU.
# Usage on poscat_gpu2:  setsid nohup reports/rl/ppo_r2_eval.sh > reports/rl/grpo_r3bc_eval/ppo_r2_driver.log 2>&1 &
cd ~/blackout-env
D=reports/rl/grpo_r3bc_eval; F=checkpoints/distill/frozen; mkdir -p $D/logs
for a in reg hlg; do
  CK=checkpoints/rl/ppo_r2_$a/final.pt
  while pgrep -f "out checkpoints/rl/ppo_r2_$a" >/dev/null || [ ! -s $CK ]; do
    pgrep -f "out checkpoints/rl/ppo_r2_$a" >/dev/null || { echo "$(date) ppo_r2_$a gone but no $CK"; exit 1; }
    sleep 120
  done
  echo "$(date) ppo_r2_$a finished"
done
sleep 30
# both GPUs must be completely free (other users' jobs come and go): wait, re-checking every 5 min
while :; do
  GPUS=""
  for g in 0 2; do nvidia-smi -i $g | grep -q "No running processes found" && GPUS="$GPUS $g"; done
  [ -n "$GPUS" ] && break
  echo "$(date) GPU 0/2 busy, waiting"; sleep 300
done
echo "$(date) using GPUs:$GPUS"
jobs=()
for a in reg hlg; do
  for o in "p24c_v321 net:$F/p24c_v321.pt" "p22_v31 net:$F/p22_v31.pt" "p5a_v22 net:$F/p5a_v22.pt" "p16_v27 net:$F/p16_v27.pt" "v32_1 V32.1"; do
    for lo in 970000 970064 970128 970192; do jobs+=("$a $o $lo $((lo + 64))"); done
  done
done
i=0
for j in "${jobs[@]}"; do echo "$j"; done | while read a ot os lo hi; do
  set -- $GPUS; n=$#; g=$(echo $GPUS | cut -d' ' -f$(( i % n + 1 ))); i=$((i + 1))
  echo "$g $a $ot $os $lo $hi"
done | xargs -P $(( $(echo $GPUS | wc -w) * 3 )) -L 1 bash -c '
  g=$0; a=$1; ot=$2; os=$3; lo=$4; hi=$5; out='$D'/ppo_r2_${a}__${ot}__${lo}.json
  [ -s $out ] && exit 0
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 30 env PYTHONPATH=. .venv/bin/python examples/eval_grpo_ckpt.py \
    --cand checkpoints/rl/ppo_r2_$a/final.pt --opp $os --lo $lo --hi $hi --out $out --device cuda:0 > '$D'/logs/ppo_r2_${a}__${ot}__${lo}.log 2>&1
  echo "$(date) done $a $ot $lo rc=$?"'
echo "$(date) ALL DONE $(ls $D/ppo_r2_*__*.json | wc -l)/40"
