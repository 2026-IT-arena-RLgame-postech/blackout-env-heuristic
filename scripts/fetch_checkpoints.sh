#!/bin/bash
# Download the frozen students, rollout nets and value nets used in docs §1–§62 from the GitHub release
# and verify them against docs/checkpoints_manifest.md5.  Run from the repo root.
set -euo pipefail
TAG=${1:-ckpt-2026-10-08}
ASSET=blackout-env-checkpoints-2026-10-08.tar
if command -v gh >/dev/null; then
  gh release download "$TAG" --repo 2026-IT-arena-RLgame-postech/blackout-env-heuristic --pattern "$ASSET" --clobber
else
  curl -L -o "$ASSET" "https://github.com/2026-IT-arena-RLgame-postech/blackout-env-heuristic/releases/download/$TAG/$ASSET"
fi
tar -xf "$ASSET" && rm -f "$ASSET"
if command -v md5sum >/dev/null; then md5sum -c docs/checkpoints_manifest.md5
else while read -r sum path; do [ "$(md5 -q "$path")" = "$sum" ] && echo "$path: OK" || { echo "$path: FAILED"; exit 1; }; done < docs/checkpoints_manifest.md5
fi
