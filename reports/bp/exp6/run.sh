#!/bin/bash
# run.sh <set> <seeds> <seed0> <workers>   (Mac)
cd /Users/mac/project/26rl/blackout-env-teach
NET=/private/tmp/claude-501/-Users-mac-project-26rl/3254af0a-9bf1-492e-98e3-b936fa522c4a/scratchpad/rl_bp16_1050.pt
C=$(PYTHONPATH=. .venv/bin/python reports/bp/exp6/c6.py $1 2>/dev/null)
PYTHONPATH=. .venv/bin/python examples/v27_eval.py play --hz 50 --ours "$C" --opponent net:$NET --seeds ${2:-12} --seed0 ${3:-90000000} --chunk 6 --cap-s 425 --stop-exhausted --workers ${4:-6} --out reports/bp/exp6/s_$1 > reports/bp/exp6/s_$1.log 2>&1
PYTHONPATH=. .venv/bin/python reports/bp/exp6/tab6.py reports/bp/exp6/s_$1
