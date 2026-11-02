#!/bin/bash
cd /Users/mac/project/26rl/blackout-env-teach
X=$(PYTHONPATH=. .venv/bin/python - <<'PY'
from blackout_env.heuristics.params import resolve_descriptor
def o(extra): return resolve_descriptor("B6,"+extra).replace("strategic_v34:","strategic_oodbp:",1)
C=[o("raid_squad=2,steal_weight=3.0,exposure_cost=2.0"),o("raid_squad=3,steal_weight=3.0,exposure_cost=2.0"),o("raid_squad=5,field_weight=0,steal_weight=3.0,exposure_cost=2.0"),o("raid_squad=3,steal_weight=3.0,exposure_cost=2.0,hunter_roles=raid/raid,raid_value=4"),o("field_weight=0.3,steal_weight=3.0"),
]
print(";".join(C))
PY
)
PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python examples/v27_eval.py play --hz 50 --ours "$X" --opponent B6 --seeds 12 --chunk 4 --cap-s 425 --stop-exhausted --workers 16 --out reports/bp/ood/screen5 > reports/bp/ood/screen5.log 2>&1
