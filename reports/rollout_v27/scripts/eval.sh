#!/bin/bash
# docs §37 Q8 evaluation on wsl_gpu (after run.sh): P16-labelled held-out sim states + agreement, then sim strength
# against the net opponents (CUDA); the heuristic opponents run on the Mac CPU (mac_eval.sh, same seeds / settings).
# usage: SEEDS=64 bash eval.sh
cd ~/blackout-env-q8
PY=.venv/bin/python
export PYTHONPATH=. OMP_NUM_THREADS=2
R=~/rv27runs; CK=$R/ck; L=$R/logs; H=$R/heldout; O=reports/rollout_v27
P16=$HOME/p14frozen/p16.pt; P14B=$HOME/p14frozen/p14b.pt; P11=$HOME/p14assets/checkpoints/distill/frozen/p11_v22.pt
V25D=$HOME/rv25runs/ck/v25d256.pt; V25F=$HOME/rv25runs/ck/v25f3x.pt; NEWD=$CK/v27d256.pt; NEWF=$CK/v27f3x.pt
mkdir -p $H $O
if [ ! -f $O/agree.json ]; then
  # held-out simulator states labelled by P16 (seeds 7200000+ / 7300000+, never trained on)
  col() { [ -f $H/$1/train.npz ] || $PY examples/rollout_net_collect.py --net $2 --teacher net:$P16 --pool "$3" --games 64 \
            --per-job 32 --workers 2 --device cuda --teacher-frac 0 --seed0 $4 --out $H/$1 > $L/heldout_$1.log 2>&1; }
  col p16 none '{"teacher": 1}' 7200000
  col self_v27d256 $NEWD '{"self": 1}' 7300000
  col self_v27f3x $NEWF '{"self": 1}' 7300000
  col self_v25d256 $V25D '{"self": 1}' 7300000
  SETS=""; for s in p16 self_v27d256 self_v27f3x self_v25d256; do SETS="$SETS $s=$H/$s/train.npz"; done
  $PY examples/rollout_v25_eval.py agree --sets $SETS --nets v27d256=small:$NEWD v27f3x=small:$NEWF v25d256=small:$V25D \
     v25f3x=small:$V25F p14b=net:$P14B --json $O/agree.json > $L/agree.log 2>&1
fi
export OMP_NUM_THREADS=1
export BLACKOUT_SMALL="v27d256=$NEWD,v27f3x=$NEWF,v25d256=$V25D,v25f3x=$V25F" BLACKOUT_NETS="p16=$P16,p14b=$P14B,p11=$P11"
$PY examples/v27_eval.py play --ours "${OURS:-p16;v27d256;v27f3x;v25d256}" --opponent "${OPPS:-p16;p14b;p11}" --seeds ${SEEDS:-64} \
   --seed0 1100000 --chunk ${CHUNK:-64} --cap-s 425 --stop-exhausted --workers ${WORKERS:-4} --device cuda --out $O/sim_wsl > $L/sim_wsl.log 2>&1
echo EVAL DONE
