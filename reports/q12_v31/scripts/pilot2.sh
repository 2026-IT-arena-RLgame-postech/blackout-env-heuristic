#!/bin/bash
# Q12 second V30-parameter sweep (docs §46.3): 32 seeds x 5 opponents (seeds 1600000+, same as pilot_params).
#   pilot2.sh a   -> base + 7 candidates (gpu2, 16 workers);  pilot2.sh b -> 4 candidates (gpu1, 8 workers; base from pilot_params)
set -eu
cd "$(dirname "$0")/../../.."
B="strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
OPPS="$B;strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve;p17;strategic_v25;strategic_v28:reserve_stage=0.5,trade_hold=advance"
if [ "${1:-a}" = a ]; then
  O="$B"; for p in threat_radius=0.10 threat_radius=0.13 threat_radius=0.20 hunter_mode=hunt reserve_count=2 replan_interval=5 sentry_radius=5.0; do O="$O;$B,$p"; done
  W=16
else
  O=""; for p in protected_storage_bonus=4.0 protected_storage_bonus=10.0 join_margin=0.15 trade_radius=2.0; do O="$O${O:+;}$B,$p"; done
  W=8
fi
exec reports/q12_v31/scripts/gate.sh reports/q12_v31/pilot2 1600000 32 ${WORKERS:-$W} "$O" "$OPPS"
