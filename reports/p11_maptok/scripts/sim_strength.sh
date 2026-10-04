#!/bin/bash
# §25 simulator strength: P11 (ref), P11 fine-tuned offline (ft_base) and + map tokens (ft_f3, ft_f2) vs V19 / P5a / V23,
# seeds 1100000-1100999 x both sides (1000 games per pairing, as §22), CUDA fp32 on GPU 2.
cd ~/blackout-env-p11m; export PYTHONPATH=$PWD BLACKOUT_MAIN=$HOME/blackout-env CUDA_VISIBLE_DEVICES=${GPU:-2}
exec ~/blackout-env/.venv/bin/python examples/rollout_net_strength.py sim --device cuda --workers ${WORKERS:-10} \
  --seeds 500 --seed0 1100000 --candidates ${CANDS:-p11 net:runs/ft_base/model.pt net:runs/ft_f3/model.pt net:runs/ft_f2/model.pt} \
  --opponents v19 p5a v23 --h2h ${H2H:-} --out runs/sim "$@"
