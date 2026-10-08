---
name: project-long-runs-detached
description: Long training/collection runs must be launched in their own session (not a harness background task) and kept on AC power
metadata: 
  node_type: memory
  type: project
  originSessionId: e938f10c-5d84-4159-8d9c-9ac7a16cbbea
  modified: 2026-09-18T16:06:26.992Z
---

Multi-hour `offline_pretrain` runs launched as a Claude Code background task die when the session ends: Run 16's first launch (2026-09-19 01:02) was killed two minutes in when the previous session closed. Launch them with `subprocess.Popen(..., start_new_session=True, stdin=DEVNULL)` (own session/process group, PPID 1) wrapped in `caffeinate -i -m -s`, and log to a file in the repo root (e.g. `run16_dagger_comp_offline_pretrain.log`).

**Why:** harness background tasks are torn down with the session; the Mac also hibernated mid-run when the charger came loose (Run 15 froze for 3 h at 1% battery), and `caffeinate -s` only prevents sleep on AC.

**How to apply:** for any run expected to outlive the conversation, launch detached as above, confirm PGID == PID and PPID 1, and remind the user to keep the charger in. Stop runs with `kill -INT <python pid>` (not the zsh wrapper). Current state of the runs is in `docs/offline_pretrain_runs.md` ("Run 14–16" section).
