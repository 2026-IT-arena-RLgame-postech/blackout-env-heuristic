#!/usr/bin/env bash
# Keep the remote GPU servers (poscat_gpu1 / poscat_gpu2) in sync with this checkout.
#
#   scripts/remote_sync.sh status            load of both servers (pick the less loaded one)
#   scripts/remote_sync.sh push [host]       code: git push the current branch into the server mirror
#                                            (~/blackout-env, receive.denyCurrentBranch=updateInstead)
#   scripts/remote_sync.sh pull [host]       results: reports/ and checkpoints/ back to the Mac (newer wins)
#   scripts/remote_sync.sh up   [host] PATH  one local data path (checkpoints/..., datasets/...) to the server
#   scripts/remote_sync.sh sync [host]       push + pull, then commit the pulled reports locally
#
# The server mirror is not a place to edit code: experiments that change code run from their own
# checkout on the server (e.g. ~/blackout-env-<task>, a clone of the mirror).  Nothing here
# deletes files on either side, and nothing goes to GitHub.
set -euo pipefail

HOSTS=(poscat_gpu1 poscat_gpu2)
REMOTE_DIR=blackout-env
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
PULLED=$(mktemp); trap 'rm -f "$PULLED"' EXIT

load() {  # "<host> <free GPU MiB total> <mean GPU util> <load1>"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" \
    'g=$(nvidia-smi --query-gpu=memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits |
         awk -F", " "{f+=\$1-\$2; u+=\$3; n++} END {printf \"%d %d\", f, u/n}");
     echo "$(hostname) $g $(cut -d" " -f1 /proc/loadavg)"' 2>/dev/null || echo "$1 unreachable"
}

pick() {  # the reachable host with the lowest load average
  for h in "${HOSTS[@]}"; do echo "$h $(load "$h")"; done |
    awk '$3 != "unreachable" {print $5, $1}' | sort -n | head -1 | cut -d" " -f2
}

host_arg() { if [[ -n "${1:-}" ]]; then echo "$1"; else pick; fi; }

push() {
  local h=$1 br
  br=$(git branch --show-current)
  git remote get-url "$h" >/dev/null 2>&1 || git remote add "$h" "$h:$REMOTE_DIR"
  # the mirror must be clean for updateInstead; it only ever holds what we pushed
  ssh "$h" "cd $REMOTE_DIR && git config receive.denyCurrentBranch updateInstead &&
            git checkout -q -- . && git checkout -q $br 2>/dev/null || git checkout -q -b $br"
  git push -q "$h" "$br"
  echo "pushed $br ($(git rev-parse --short HEAD)) to $h:$REMOTE_DIR"
  [[ -z "$(git status --porcelain --untracked-files=no)" ]] ||
    echo "note: uncommitted local changes were not sent (commit them first)"
}

pull() {
  local h=$1
  rsync -a --update --exclude '.DS_Store' --out-format='reports/%n' "$h:$REMOTE_DIR/reports/" reports/ |
    grep -v '/$' > "$PULLED" || true
  rsync -a --update --exclude '.DS_Store' "$h:$REMOTE_DIR/checkpoints/" checkpoints/
  echo "pulled reports/ and checkpoints/ from $h"
}

up() {
  local h=$1 p=$2
  ssh "$h" "mkdir -p $REMOTE_DIR/$(dirname "$p")"
  rsync -a --update "$p" "$h:$REMOTE_DIR/$(dirname "$p")/"
  echo "sent $p to $h"
}

commit_reports() {  # only the report files this pull brought (<20 MB each) + tracked docs/reports edits
  while read -r f; do
    [[ -f "$f" && $(wc -c < "$f") -lt 20000000 ]] && git add -- "$f"
  done < "$PULLED"
  git add -u reports docs
  if git diff --cached --quiet; then echo "nothing new to commit"; return; fi
  git commit -q -m "reports: sync results from $1" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  echo "committed $(git rev-parse --short HEAD)"
}

cmd=${1:-status}; shift || true
case "$cmd" in
  status) for h in "${HOSTS[@]}"; do load "$h"; done | awk '{print $1, "freeGPU", $2 "MiB", "util", $3 "%", "load", $4}'
          echo "less loaded: $(pick)" ;;
  push) push "$(host_arg "${1:-}")" ;;
  pull) pull "$(host_arg "${1:-}")" ;;
  up) [[ $# -ge 2 ]] || { echo "usage: up HOST PATH"; exit 2; }; up "$1" "$2" ;;
  sync) h=$(host_arg "${1:-}"); push "$h"; pull "$h"; commit_reports "$h" ;;
  *) sed -n 2,14p "$0"; exit 2 ;;
esac
