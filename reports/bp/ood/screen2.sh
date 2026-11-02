#!/bin/bash
cd /Users/mac/project/26rl/blackout-env-teach
C=(
"B6,hunter_roles=east/east,roles_when=always"
"B6,hunter_roles=default/default,roles_when=always"
"B6,hunter_roles=raid/south,roles_when=always"
"B6,hunter_roles=,hunt_hh=30,guard_fill=false"
"B6,carrier_quota=0"
"B6,carrier_delay=40"
"mixcls:1:B6|idle"
"mixcls:02:B6|idle"
"mixcls:1:B6|random"
"mixcls:02:B6|random"
"perturb:eps=0.03:B6"
"perturb:lag=2:B6"
"perturb:freeze=0.04:B6"
"switch:25:B6|perturb:eps=0.15:B6"
"switch:25:B6|perturb:lag=6:B6"
"switch:25:B6|perturb:freeze=0.3:B6"
"switch:30:B6|B6,hunter_roles=south/east,roles_when=always"
"switch:30:B6|B6,hunter_roles=,solo_sentry=true,sentry_min_hunters=1,sentry_offset=1.5,sentry_radius=10,guard_fill=true"
"B6,steal_weight=3.0,exposure_cost=0.3,hunter_roles=raid/raid,raid_value=4,raid_carrier=0.5"
"B6,storm_units=3,storm_after=4,hunter_quota=1"
"V34a"
"V34b"
"B3"
"B3R"
"B5"
"B4"
)
J=$(IFS=';'; echo "${C[*]}")
PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python examples/v27_eval.py play --hz 50 --ours "$J" --opponent B6 --seeds 8 --chunk 4 --cap-s 425 --stop-exhausted --workers 16 --out reports/bp/ood/screen2 > reports/bp/ood/screen2.log 2>&1
