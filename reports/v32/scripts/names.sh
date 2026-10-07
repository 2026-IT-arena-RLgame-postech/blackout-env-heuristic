# V32 (docs §58) descriptor aliases, sourced by gate.sh; ``expand "a;b"`` maps alias names to descriptors.
F=${R:-.}/checkpoints/distill/frozen
export BLACKOUT_NETS=p22=$F/p22_v31.pt,p21e=$F/p21e_v30.pt,p20=$F/p20_v30.pt,p17=$F/p17_v28.pt,p14b=$F/p14_v25.pt
P="reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
T31="threat_radius=0.13,camp_engage_radius=6.5"
declare -A N=(
  [V31]="strategic_v30:$P,$T31" [V30]="strategic_v30:$P"
  [V29]="strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve"
  [V28]="strategic_v28:reserve_stage=0.5,trade_hold=advance" [V27]="strategic_v27:reserve_stage=1.5"
  [V25]="strategic_v25" [V23]="strategic_v23"
  [H4]="strategic_v30:$P,hunter_quota=4" [H4HUNT]="strategic_v30:$P,hunter_quota=4,hunter_mode=hunt"
  [A]="strategic_v32:$P,$T31,hunter_quota=4" [A0]="strategic_v32:$P,hunter_quota=4"
)
expand() {
  local out=() x
  IFS=';' read -ra xs <<< "$1"
  for x in "${xs[@]}"; do
    local head=${x%%,*} rest=""
    [[ "$x" == *,* ]] && rest=",${x#*,}"
    if [[ -n "${N[$head]:-}" ]]; then out+=("${N[$head]}$rest"); else out+=("$x"); fi
  done
  (IFS=';'; echo "${out[*]}")
}
