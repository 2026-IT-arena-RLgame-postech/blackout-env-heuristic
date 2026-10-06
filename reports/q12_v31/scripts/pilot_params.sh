#!/bin/bash
# Q12 existing-parameter sweep on top of V30 (docs §46.3): 32 seeds x 5 opponents, base = V30.
set -eu
cd "$(dirname "$0")/../../.."
B="strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
OURS="$B"
for p in carrier_quota=0 carrier_quota=2 exposure_cost=0.5 exposure_cost=1.4 danger_tiles=2.0 danger_tiles=4.5 \
         camp_engage_radius=3.0 camp_engage_radius=6.5 sentry_offset=1.5 sentry_offset=3.5 risk_weight=5.0; do
  OURS="$OURS;$B,$p"
done
OPPS="$B;strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve;p17;strategic_v25;strategic_v28:reserve_stage=0.5,trade_hold=advance"
exec reports/q12_v31/scripts/gate.sh reports/q12_v31/pilot_params 1600000 32 ${WORKERS:-8} "$OURS" "$OPPS"
