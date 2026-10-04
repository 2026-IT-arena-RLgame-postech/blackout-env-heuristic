#!/bin/bash
# docs §39 Q9 on wsl_gpu (after run.sh): P17-labelled held-out sim states + agreement with P17.
# Sim strength runs on the Mac CPU (mac_eval.sh; wsl CUDA sim is slow when the GPU is shared, §37.4).
cd ~/blackout-env-q9
PY=.venv/bin/python
export PYTHONPATH=. OMP_NUM_THREADS=2
R=~/rv28runs; CK=$R/ck; L=$R/logs; H=$R/heldout; O=reports/rollout_v28
P17=$HOME/p14frozen/p17.pt; P16=$HOME/p14frozen/p16.pt; P14B=$HOME/p14frozen/p14b.pt
V27D=$HOME/rv27runs/ck/v27d256.pt; V27F=$HOME/rv27runs/ck/v27f3x.pt; NEWD=$CK/v28d256.pt; NEWF=$CK/v28f3x.pt
mkdir -p $H $O
# held-out simulator states labelled by P17 (seeds 7400000+ / 7500000+, never trained on)
col() { [ -f $H/$1/train.npz ] || $PY examples/rollout_net_collect.py --net $2 --teacher net:$P17 --pool "$3" --games 64 \
          --per-job 32 --workers 2 --device cuda --teacher-frac 0 --seed0 $4 --out $H/$1 > $L/heldout_$1.log 2>&1; }
col p17 none '{"teacher": 1}' 7400000
col self_v28d256 $NEWD '{"self": 1}' 7500000
col self_v28f3x $NEWF '{"self": 1}' 7500000
col self_v27d256 $V27D '{"self": 1}' 7500000
SETS=""; for s in p17 self_v28d256 self_v28f3x self_v27d256; do SETS="$SETS $s=$H/$s/train.npz"; done
$PY examples/rollout_v25_eval.py agree --sets $SETS --nets v28d256=small:$NEWD v28f3x=small:$NEWF v27d256=small:$V27D \
   v27f3x=small:$V27F p16=net:$P16 --json $O/agree.json > $L/agree.log 2>&1
echo EVAL DONE
