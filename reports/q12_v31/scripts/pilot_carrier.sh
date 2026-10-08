#!/bin/bash
# Q12 Carrier-evasion pilot (docs §46.3): 32 seeds x 5 opponents, base = V30.
set -eu
cd "$(dirname "$0")/../../.."
K="reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
B="strategic_v30:$K"
OURS="$B"
for p in carrier_col_scale=0.0 carrier_col_scale=0.5 carrier_hunter_scale=0.6 carrier_col_scale=0.0,carrier_hunter_scale=0.6; do
  OURS="$OURS;strategic_v31:$K,$p"
done
OURS="$OURS;$B,exact_evasion=true"
OPPS="$B;strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve;p17;strategic_v25;strategic_v28:reserve_stage=0.5,trade_hold=advance"
exec reports/q12_v31/scripts/gate.sh reports/q12_v31/${OUT:-pilot_carrier} ${SEED0:-1600000} ${SEEDS:-32} ${WORKERS:-8} "$OURS" "$OPPS"
