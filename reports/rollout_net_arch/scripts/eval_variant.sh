#!/bin/bash
# eval one trained variant: self-state agreement (net x net sim, V23 labels), held-out rows, sim strength
cd ~/blackout-env-arch
PY=~/blackout-env/.venv/bin/python
export PYTHONPATH=. OMP_NUM_THREADS=1 BLACKOUT_MAIN=$HOME/blackout-env
V=$1; NET=${NET:-checkpoints/rollout_net_arch/$V.pt}; W=${WORKERS:-8}; SEEDS=${SEEDS:-500}
R=reports/rollout_net_arch; SS=checkpoints/rollout_net_arch/selfstate_$V
mkdir -p $R/heldout $R/strength/$V
if [ ! -f $R/heldout/${V}_full.json ]; then
  $PY examples/rollout_net_collect.py --net $NET --games 64 --per-job 16 --workers 4 --seed0 1300000 --teacher-frac 0 \
     --pool '{"small": 1.0}' --cap-s 150 --out $SS > $R/heldout/${V}_selfstate.log 2>&1
  $PY examples/rollout_net_train.py eval --data checkpoints/rollout_net/data --net $NET \
     --val2 checkpoints/rollout_net/simval/val.npz $SS/train.npz --json $R/heldout/${V}_full.json >> $R/heldout/${V}_selfstate.log 2>&1
  rm -rf $SS
fi
S=$R/strength/$V
[ -f $S/games.jsonl ] || grep -E '"candidate": "(p11|r5)", "opponent": "(v19|p5a|v23)"' reports/rollout_net_strength/sim/games.jsonl > $S/games.jsonl
$PY examples/rollout_net_strength.py sim --candidates small:$NET --opponents v19 p5a v23 --h2h --seeds $SEEDS \
   --workers $W --device cuda --out $S > $S/stdout.log 2>&1
$PY examples/rollout_net_strength.py summary --ref p11 --out $S > /dev/null 2>&1
cp $S/summary.txt $S/summary_vs_p11.txt
$PY examples/rollout_net_strength.py summary --ref r5 --out $S > /dev/null 2>&1
cp $S/summary.txt $S/summary_vs_r5.txt
echo EVAL DONE $V
