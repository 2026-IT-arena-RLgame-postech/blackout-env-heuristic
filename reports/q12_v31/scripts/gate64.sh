#!/bin/bash
# Q12 V31 gate, stage 64 seeds (docs §46.4): fresh seeds 1700000+, pool 10 (V30 h2h + 8 + held-out V30(0.75)).
#   gate64.sh a -> base + 3 candidates (gpu2, 16 workers);  gate64.sh b -> combo (gpu1, 8 workers)
set -eu
cd "$(dirname "$0")/../../.."
K="reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
B="strategic_v30:$K"
POOL="$B;strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve;strategic_v28:reserve_stage=0.5,trade_hold=advance;p17;p16;strategic_v27:reserve_stage=1.5;strategic_v25;p14b;strategic_v23;strategic_v30:reserve_stage=0.75,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
if [ "${1:-a}" = a ]; then
  O="$B;$B,threat_radius=0.13;$B,camp_engage_radius=6.5;$B,sentry_radius=8.0"; W=16
else
  O="$B,threat_radius=0.13,camp_engage_radius=6.5"; W=8
fi
exec reports/q12_v31/scripts/gate.sh reports/q12_v31/${OUT:-gate64} ${SEED0:-1700000} ${SEEDS:-64} ${WORKERS:-$W} "$O" "$POOL"
