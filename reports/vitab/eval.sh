#!/bin/bash
# ViT-encoder A/B (docs/rl_finetune_plan_ko.md "36토큰 ViT 인코더") strength gate, work unit 3.  Runs in ~/blackout-env-vitab on poscat_gpu2.
# Six students (A/B1/B2 x seed 0/1) vs p24c_v321, p22_v31, V19 (strategic_v19), P5a (p5a_v22.pt) + head-to-head across arms; same seeds
# and sides for everyone (v27_eval.py play: full sim matches, cap 425 s, --stop-exhausted).  Nets on the given GPU.
# Usage: GPU=0 SEEDS=64 WORKERS=8 reports/vitab/eval.sh [opp ...]    (default: all opponents; names: p24c p22 p5a v19 h2h)
R=$HOME/blackout-env-vitab; cd $R; OUT=${OUT:-reports/vitab/sim}; mkdir -p $OUT
export PYTHONPATH=$R NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${GPU:-0}
V=checkpoints/distill/vitab; F=checkpoints/distill/frozen
export BLACKOUT_NETS=""
for a in A B1 B2; do for s in 0 1; do BLACKOUT_NETS="$BLACKOUT_NETS,${a}s$s=$R/$V/${a}_s$s/final.pt"; done; done
export BLACKOUT_NETS="${BLACKOUT_NETS#,},p24c=$R/$F/p24c_v321.pt,p22=$R/$F/p22_v31.pt,p5a=$R/$F/p5a_v22.pt"
ALL="As0;As1;B1s0;B1s1;B2s0;B2s1"
SEEDS=${SEEDS:-64}; SEED0=${SEED0:-5800000}
play() { # ours opponent
  nice -n 30 .venv/bin/python examples/v27_eval.py play --seeds $SEEDS --seed0 $SEED0 --cap-s 425 --stop-exhausted --device cuda \
    --chunk ${CHUNK:-16} --workers ${WORKERS:-8} --out $OUT --ours "$1" --opponent "$2"
}
OPPS=${*:-"p24c p22 p5a v19 h2h"}
for o in $OPPS; do
  case $o in
    p24c|p22) SEEDS=${SEEDS_NET:-32} play "$ALL" $o ;;  # far stronger opponents (students lose by ~75): 32 seeds suffice and save wall time
    p5a) play "$ALL" $o ;;
    v19) play "$ALL" strategic_v19 ;;
    h2h) # arm-vs-arm on the same seeds: A vs B1, A vs B2, B1 vs B2 (training-seed-0 pairs only, to save wall time)
      for s in 0; do play "As$s" B1s$s; play "As$s" B2s$s; play "B1s$s" B2s$s; done ;;
  esac
done
echo "EVAL DONE $OPPS" >> $OUT/stdout.log
