#!/usr/bin/env bash
# Keep the remote GPU servers (poscat_gpu1 / poscat_gpu2) in sync with this checkout.
#
#   scripts/remote_sync.sh status            load of both servers (pick the less loaded one)
#   scripts/remote_sync.sh push [host]       code: git push the current branch into the server mirror
#                                            (~/blackout-env, receive.denyCurrentBranch=updateInstead)
#   scripts/remote_sync.sh pull [host]       results: reports/ and checkpoints/ back to the Mac (newer wins)
#   scripts/remote_sync.sh up   [host] PATH  one local data path (checkpoints/..., datasets/...) to the server
#   scripts/remote_sync.sh sync [host]       push + pull, then commit the pulled reports locally
#   scripts/remote_sync.sh procs [host]      my processes on the server(s); orphans (parent = init/systemd) flagged
#   scripts/remote_sync.sh kill-orphans [host]  kill my orphaned python / BlackOut processes (asks first)
#   scripts/remote_sync.sh clean [host]      delete my temp files: /tmp files >1 h old, Unity Player logs,
#                                            __pycache__ / .pytest_cache, torch inductor caches >1 day old
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
            git checkout -q -- . && { git checkout -q $br 2>/dev/null || git checkout -q -b $br; }"
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

procs() {  # pid ppid elapsed rss(MB) %cpu command; ORPHAN = reparented to init/systemd (user or system)
  ssh "$1" 'for p in $(pgrep -u "$USER" -f "python|BlackOut" | grep -v "^$$\$"); do
      pp=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d " ") || continue
      pc=$(ps -o comm= -p "$pp" 2>/dev/null)
      tag=""; [[ "$pp" == 1 || "$pc" == systemd ]] && tag=ORPHAN
      ps -o pid=,etime=,rss=,pcpu=,args= -p "$p" | awk -v t="$tag" "{\$3=int(\$3/1024)\"M\"; c=\$0; print t, substr(c,1,150)}"
    done' 2>/dev/null
}

kill_orphans() {
  local list
  list=$(procs "$1" | awk '$1 == "ORPHAN" {print $2}')
  [[ -n "$list" ]] || { echo "no orphans on $1"; return; }
  procs "$1" | grep '^ORPHAN'
  read -r -p "kill these on $1? [y/N] " a; [[ "$a" == y ]] || return
  ssh "$1" "kill $list; sleep 3; kill -9 $list 2>/dev/null; true"
  echo "killed: $list"
}

clean() {
  ssh "$1" 'before=$(df --output=avail -BG ~ | tail -1)
    find /tmp /var/tmp -maxdepth 2 -user "$USER" -mmin +60 \( -type f -o -type s \) -delete 2>/dev/null
    find /tmp -maxdepth 1 -user "$USER" -mmin +60 -type d -empty -delete 2>/dev/null
    rm -rf ~/.config/unity3d/*/*/Player*.log 2>/dev/null
    find ~/blackout-env* -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
    find ~/blackout-env* -name .pytest_cache -type d -prune -exec rm -rf {} + 2>/dev/null
    find /tmp/torchinductor_$USER ~/.cache/torch/inductor -mindepth 1 -maxdepth 1 -mtime +1 -exec rm -rf {} + 2>/dev/null
    echo "$(hostname): free ${before} -> $(df --output=avail -BG ~ | tail -1)"; du -sh ~ 2>/dev/null'
}

for_hosts() {  # run $1 on the given host, or on every host
  if [[ -n "${2:-}" ]]; then "$1" "$2"; else for h in "${HOSTS[@]}"; do echo "== $h"; "$1" "$h"; done; fi
}

cmd=${1:-status}; shift || true
case "$cmd" in
  status) for h in "${HOSTS[@]}"; do load "$h"; done | awk '{print $1, "freeGPU", $2 "MiB", "util", $3 "%", "load", $4}'
          echo "less loaded: $(pick)" ;;
  push) push "$(host_arg "${1:-}")" ;;
  pull) pull "$(host_arg "${1:-}")" ;;
  up) [[ $# -ge 2 ]] || { echo "usage: up HOST PATH"; exit 2; }; up "$1" "$2" ;;
  sync) h=$(host_arg "${1:-}"); push "$h"; pull "$h"; commit_reports "$h" ;;
  procs) for_hosts procs "${1:-}" ;;
  kill-orphans) for_hosts kill_orphans "${1:-}" ;;
  clean) for_hosts clean "${1:-}" ;;
  *) sed -n 2,18p "$0"; exit 2 ;;
esac
