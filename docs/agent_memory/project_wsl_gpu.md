---
name: project-wsl-gpu
description: "wsl_gpu = user's Windows laptop WSL2 (RTX 3060 Laptop 6 GB) over Tailscale; repo mirror + venv identical to the poscat servers"
metadata:
  node_type: memory
  type: project
  originSessionId: f971541c-d34c-47d7-b925-80e38fb1e2ec
  modified: 2026-10-03T17:09:03.830Z
---

Set up 2026-10-04 at the user's request. Another remote host next to [[project-remote-gpu-servers]].

- Access: `ssh wsl_gpu` (~/.ssh/config: Tailscale host, key auth — details in your own ~/.ssh/config). Machine name `laptop`.
- Hardware: Ubuntu 24.04 on WSL2, 12 cores, 23 GB RAM (`.wslconfig` memory=24GB), 150 GB VHDX (C: has ~212 GB free). GPU is an RTX 3060 Laptop, 6 GB, sm_86 (bf16 OK), ~12 fp16 TFLOPS. Good for small jobs and debugging, not for big training.
- `~/.bashrc` line 1 puts `/usr/lib/wsl/lib` (nvidia-smi/libcuda) and `~/.local/bin` (uv) on PATH. Without it, non-interactive ssh can't find nvidia-smi.
- Repo mirror `~/blackout-env`: `scripts/remote_sync.sh push|pull wsl_gpu` works. It is not in the script's HOSTS, so status/pick ignore it.
- `.venv` was built with uv from gpu1's exact pip freeze (`build/requirements-gpu1.txt`, `--no-deps`): torch 2.11.0+cu128, protobuf C++ wheel copied from gpu1, `api_implementation.Type()=='cpp'`. Full test suite passes, including torch.compile/inductor (python3.12-dev installed 2026-10-04).
- Not copied yet: Unity Linux build, checkpoints, datasets.
- It's a laptop and may sleep or reboot. Check reachability first and use tmux/nohup for runs.
