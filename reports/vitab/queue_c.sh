#!/bin/bash
# Stride-2 ViT arms (C1 patch2, C2 patch4s2, C3 patch3s2; docs/rl_finetune_plan_ko.md) with the same recipe/data as the
# A/B (train.sh).  Two lanes: GPU 0 takes the jobs in order, GPU 1 (another user's) takes them in reverse order but only
# while it is completely empty.  Before EVERY start a lane re-checks nvidia-smi: no compute process and no graphics (G / C+G)
# process on its GPU and < 200 MiB used; otherwise it waits.  GPUs 2 and 3 are never touched.
# Usage (detached by the caller): reports/vitab/queue_c.sh "C2:patch4s2:0 C1:patch2:0 ..."   log: $LOG
R=${R:-$HOME/blackout-env-vitabc}; BASE=$HOME/blackout-env-vitab; cd $R
LOG=${LOG:-$BASE/checkpoints/distill/vitab/queue_c.log}
export SHARDS=$BASE/checkpoints/distill/vitab/data/shards OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
JOBS=($1)
gate() {  # $1 = gpu index; 0 = free
  local procs used
  procs=$(nvidia-smi | awk -v g=$1 '/^\| +[0-9]+ +N\/A +N\/A/ && $2==g')
  [ -z "$procs" ] || { echo "$(date) gate gpu$1: busy: $(echo "$procs" | awk '{print $5}' | tr '\n' ' ')"; return 1; }
  used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g{print $2}')
  [ "$used" -lt 200 ] || { echo "$(date) gate gpu$1: used $used MiB"; return 1; }
}
lane() {  # $1 = gpu, $2 = order (fwd|rev)
  local gpu=$1 list=("${JOBS[@]}") job arm enc seed out
  [ $2 = rev ] && list=($(printf '%s\n' "${JOBS[@]}" | tac))
  for job in "${list[@]}"; do
    IFS=: read arm enc seed <<< "$job"; out=$BASE/checkpoints/distill/vitab/${arm}_s${seed}
    [ -f $out/result.json ] && continue
    while :; do
      [ -d $out/.claim ] && continue 2          # another lane has it
      if gate $gpu >> $LOG; then mkdir -p $out && mkdir $out/.claim 2>/dev/null && break; fi
      [ $gpu != 0 ] && [ $(ls -d $BASE/checkpoints/distill/vitab/*/.claim 2>/dev/null | wc -l) -ge ${#JOBS[@]} ] && return
      sleep 120
    done
    echo "$(date) start $job on gpu$gpu" >> $LOG
    CUDA_VISIBLE_DEVICES=$gpu nice -n 30 env PYTHONPATH=. R=$R reports/vitab/train.sh $enc $seed $out > $out/run.log 2>&1
    echo "$(date) done $job rc=$? (gpu$gpu)" >> $LOG
  done
}
lane 0 fwd &
lane 1 rev &
wait
