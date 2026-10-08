---
name: project-remote-gpu-servers
description: Run heavy sim/training on remote servers poscat_gpu1 / poscat_gpu2 (pick the less loaded one); Unity evals too (Linux server build)
metadata:
  type: project
---
User asked (2026-10-03) to move work off the MacBook (screen overheating) to SSH servers `poscat_gpu1` / `poscat_gpu2` (~/.ssh/config, same IP, ports 19721/19722, key auth set up). Each: Ubuntu 24.04, 28 cores, 62 GB RAM, 4× RTX 2080 Ti 11 GB (sm_75: fp16 ok, no bf16), shared with other users (gpu1 had 17 logins), disk ~90% full.

**How to apply:**
- Before launching, check both (`nvidia-smi`, `uptime`, `free -g`) and use the one with more free GPU/CPU; pick idle GPUs via CUDA_VISIBLE_DEVICES; don't hog a shared box.
- Repo at ~/blackout-env (rsync from Mac, .venv with CUDA torch cu128, `pip install -e . --no-deps`); checkpoints rsynced selectively. No GitHub push without asking.
- Unity evals now run on the servers too (2026-10-03, docs §21): Linux server build is bit-identical to the Mac build (open-loop replay), Mac evals reproduce per match. Use ~12 Unity workers per server (shared boxes; 24 gives nothing). Server venvs need the locally built protobuf C++ wheel (`~/blackout-env/build/wheels/protobuf-3.20.3-cp312-*.whl`, check `api_implementation.Type()=='cpp'`) or Unity round-trips are 3x slower.
- Long runs: `nohup`/tmux on the server (no caffeinate needed). Related: [[project-long-runs-detached]].
- Hygiene (user, 2026-10-03; disk is nearly full): delete temp files often (`scripts/remote_sync.sh clean`), never leave orphaned workers (`procs` flags ORPHAN, `kill-orphans`); launch jobs so children die with the parent (process group / `setsid` + kill the group, Unity envs closed in `finally`), and check `procs` after every run or abort. Delete intermediate data (shards, records) once consumed or rsynced back.
- Unity on servers: Linux Dedicated Server build `build/linux_server/BlackOut.x86_64` (blackout CIBuild.BuildBlackOutLinuxServer, Unity 6000.4.11f1 with Linux modules installed 2026-10-03; grpc plugin copied from Plugins/AnyCPU by the build script). Connects in ~1 s.

**2026-10-04: poscat_gpu1 GPUs are OFF-LIMITS (another user). Later that day the user allowed CPU-only jobs on gpu1 as long as they don't disturb the other user: no CUDA, `nice -n 10`, ~8 workers max (gpu1 had load ~7–14/28 then), check load before launching.** GPU work → gpu2 only. (Earlier the same day gpu2 was briefly handed back, then the user moved it the other way.) Ignore `remote_sync.sh status`'s "less loaded" pick while this holds. gpu1↔gpu2 cannot ssh each other; copy data through the Mac: `ssh poscat_gpu1 'tar cf - X' | ssh poscat_gpu2 'tar xf -'`.

- Gotcha (2026-10-06): ssh sessions on poscat_gpu2 start at nice -20, so `nice -n 10` gives -10 (above other users). Use `nice -n 30` or `renice -n 10` after launch, and check with `ps -o ni`.
- User OK (2026-10-06): extra CPU jobs on gpu2 are fine at nice 19 (other users' load comes in bursts). Use `nice -n 39`, since sessions start at -20.
- User OK (2026-10-07): gpu1 CPU may go to ~12 processes total (the usual 8 plus 4 more), still nice 10 and CPU-only. Check load first (other users ~5 cores then).
- User (2026-10-08): gpu2 CPU is overloaded by other users (load ~165) → move CPU work to gpu1 (up to ~12 workers, nice 10); keep gpu2 for GPU search processes only.
