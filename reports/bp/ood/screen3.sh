#!/bin/bash
cd /Users/mac/project/26rl/blackout-env-teach
X=$(PYTHONPATH=. .venv/bin/python - <<'PY'
from blackout_env.heuristics.params import resolve_descriptor
def o(extra): return resolve_descriptor("B6,"+extra).replace("strategic_v34:","strategic_oodbp:",1)
C=[o("hunter_plan=0:2/25:0/240:2,hide=home"),o("hunter_plan=0:2/25:0/150:2,hide=home"),o("hunter_plan=0:2/25:0,hide=home"),o("hunter_plan=0:2/25:0,hide=stay"),o("hunter_plan=0:2/25:0/330:2,hide=home"),
"B6,hunter_roles=,hunt_hh=30,guard_fill=false,hunter_quota=2,exit_guards=0",
"B6,hunter_roles=raid/raid,roles_when=always,raid_value=4,raid_carrier=4",
"B6,steal_weight=3.0,exposure_cost=0.3,carrier_quota=0",
"B6,sentry_offset=9,hunter_roles=",
"B6,hunter_roles=south/raid,roles_when=always,bait=6",
]
print(";".join(C))
PY
)
PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python examples/v27_eval.py play --hz 50 --ours "$X" --opponent B6 --seeds 8 --chunk 4 --cap-s 425 --stop-exhausted --workers 16 --out reports/bp/ood/screen3 > reports/bp/ood/screen3.log 2>&1
