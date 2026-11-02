#!/bin/bash
cd /Users/mac/project/26rl/blackout-env-teach
X=$(PYTHONPATH=. .venv/bin/python - <<'PY'
from blackout_env.heuristics.params import resolve_descriptor
def o(extra): return resolve_descriptor("B6,"+extra).replace("strategic_v34:","strategic_oodbp:",1)
C=[o("hunter_plan=0:2/12:0/40:2,hide=home"),o("hunter_plan=0:2/12:0/60:2,hide=home"),o("hunter_plan=0:2/12:0,hide=home"),o("hunter_plan=0:2/12:0/50:2,hide=stay"),o("hunter_plan=0:2/20:0/50:2"),
"B6,sentry_offset=9,hunter_roles=,guard_fill=true,guard_after=0",
"B6,sentry_offset=7,sentry_radius=10,hunter_roles=default/raid,roles_when=always",
"B6,hunt_hh=30,hunter_roles=raid/raid,roles_when=always,raid_value=4",
]
print(";".join(C))
PY
)
PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python examples/v27_eval.py play --hz 50 --ours "$X" --opponent B6 --seeds 8 --chunk 4 --cap-s 425 --stop-exhausted --workers 16 --out reports/bp/ood/screen4 > reports/bp/ood/screen4.log 2>&1
