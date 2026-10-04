#!/bin/bash
# docs §31 evaluation chain on wsl_gpu (after both chains): held-out sets + agreement, rollout cost, sim strength
cd ~/blackout-env-rv25
PY=.venv/bin/python
export PYTHONPATH=. OMP_NUM_THREADS=2 BLACKOUT_MAIN=$HOME/rv25main
R=~/rv25runs; CK=$R/ck; L=$R/logs; H=$R/heldout; O=reports/rollout_v25
P14B=$HOME/p14frozen/p14b.pt; P11=$HOME/p14assets/checkpoints/distill/frozen/p11_v22.pt
F3X=$HOME/rv25assets/ck/f3_x_d2.pt; D256=$HOME/rv25assets/ck/d256_f3_x32_d2.pt; R5=$HOME/rv25assets/ck/r5.pt
NEWF=$CK/v25f3x.pt; NEWD=$CK/v25d256.pt
mkdir -p $H $O
# 1. held-out simulator states labelled by P14b (seeds 7000000+, never trained on): P14b self-play, each net's self-play
col() { [ -f $H/$1/train.npz ] || $PY examples/rollout_net_collect.py --net $2 --teacher net:$P14B --pool "$3" --games 64 \
          --per-job 32 --workers 2 --device cuda --teacher-frac 0 --seed0 $4 --out $H/$1 > $L/heldout_$1.log 2>&1; }
col p14b none '{"teacher": 1}' 7000000
col self_v25f3x $NEWF '{"self": 1}' 7100000
col self_v25d256 $NEWD '{"self": 1}' 7100000
col self_f3x $F3X '{"self": 1}' 7100000
col self_d256 $D256 '{"self": 1}' 7100000
for s in p14b self_v25f3x self_v25d256 self_f3x self_d256; do SETS="$SETS $s=$H/$s/train.npz"; done
$PY examples/rollout_v25_eval.py agree --sets $SETS --nets v25f3x=small:$NEWF v25d256=small:$NEWD f3x=small:$F3X d256x32=small:$D256 \
   r5=small:$R5 p11=net:$P11 --json $O/agree.json > $L/agree.log 2>&1
# 2. rollout cost: the whole BatchRollout loop (render + nets + step), mid-game roots, alternating rounds
for round in 1 2; do for n in r5:$R5 f3x:$F3X v25f3x:$NEWF d256x32:$D256 v25d256:$NEWD; do
  $PY examples/batch_rollout_bench.py throughput --policy small:${n#*:} --configs cuda:fp32 cuda:fp16 --sizes 256 2048 \
     --threads 6 --min-decisions 200000 --tag _${n%%:*}_r$round --out $O/throughput >> $L/throughput.log 2>&1
done; done
# 3. strength: heuristic opponents 128 seeds x 2 sides, net opponents 256 x 2 (seeds 1100000+), paired vs f3x
export EXTRA_CKPTS="v25f3x=$NEWF,v25d256=$NEWD"
CANDS="f3x v25f3x v25d256 d256x32 p14b" OPPS="v23 v25" SEEDS=128 CHUNK=64 WORKERS=6 OUT=$O/sim bash $O/scripts/eval.sh > $L/sim_heur.log 2>&1
CANDS="f3x v25f3x v25d256 d256x32 p14b" OPPS="p11 p14b" H2H="p14b,p14b" SEEDS=256 CHUNK=128 WORKERS=4 OUT=$O/sim bash $O/scripts/eval.sh > $L/sim_net.log 2>&1
echo EVAL DONE
