#!/bin/bash
# screen 1: candidate personalities vs B6 (candidate = "ours")
cd /Users/mac/project/26rl/blackout-env-teach
C=(
"B6,hunter_quota=0"
"B6,hunter_quota=1"
"strategic_early_rush"
"strategic_fortress"
"strategic_raider"
"strategic_duelist"
"strategic_cargo_hunter"
"B6,hunter_roles=south/east,roles_when=always"
"B6,hunter_roles=south/south,roles_when=always"
"B6,hunter_roles=,solo_sentry=true,sentry_min_hunters=1,sentry_offset=1.5,sentry_radius=10,guard_fill=true"
"B6,steal_weight=3.0,exposure_cost=0.3"
"B6,hunter_roles=,exit_guards=2,guard_radius=10,exposure_cost=2.0,steal_weight=0.8,guard_fill=false"
"B6,storm_units=3,storm_after=4"
"B6,bait=6"
"V34c"
"V33bp"
"random"
"idle"
"perturb:eps=0.1:B6"
"perturb:lag=4:B6"
"perturb:freeze=0.15:B6"
"switch:30:B6,hunter_quota=0|B6"
"switch:30:B6|B6,hunter_quota=0"
)
J=$(IFS=';'; echo "${C[*]}")
PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python examples/v27_eval.py play --hz 50 --ours "$J" --opponent B6 --seeds 8 --chunk 4 --cap-s 425 --stop-exhausted --workers 16 --out reports/bp/ood/screen1 > reports/bp/ood/screen1.log 2>&1
