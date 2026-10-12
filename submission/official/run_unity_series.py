"""The submission in the official runner: ``BlackOutEnv`` + ``run_series`` on the official build.

Uses only the official package (ist-tech-AI-games/blackout-env on PYTHONPATH) and the submission
files.  Each side is a ``CheckpointModel`` from ``load_checkpoint``; ours is wrapped to cross-check
every tick the tracker's reconstruction against what each of its 5 agents' official obs shows
(own class, everyone's held item type, both scores).

    PYTHONPATH=<official blackout-env> python run_unity_series.py --sub <dir> --build <BlackOut.x86_64> \\
        --opp self|readme --matches 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time

import numpy as np
import torch
import torch.nn as nn
from blackout_env import BlackOutEnv, BaseModel, load_checkpoint, run_series


class ReadmePolicy(nn.Module):
    """README's example MyPolicy (random weights): a baseline that moves."""

    def __init__(self, vector_size: int, n_channels: int):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(n_channels, 16, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)))
        self.mlp = nn.Sequential(nn.Linear(32 * 4 * 4 + vector_size, 256), nn.ReLU(), nn.Linear(256, 2), nn.Tanh())

    def forward(self, vector, graphic):
        return self.mlp(torch.cat([vector, self.cnn(graphic).flatten(1)], dim=1))


class Checked(BaseModel):
    """Our CheckpointModel, timed and cross-checked against the official obs every call."""

    def __init__(self, inner):
        self.inner, self.bad, self.calls, self.t, self.worst = inner, 0, 0, 0.0, 0.0

    def act(self, obs):
        t0 = time.perf_counter()
        out = self.inner.act(obs)
        dt = time.perf_counter() - t0
        self.t += dt
        self.worst = max(self.worst, dt) if self.calls else 0.0
        self.calls += 1
        pol = self.inner._net
        tr = pol._tracker
        if tr is not None:
            sim = tr.sim
            for k, (name, o) in enumerate(obs.items()):
                u = int(name.split("_")[1])
                v = o["vector"]
                if int(np.argmax(v[90:93])) != int(sim.ui[u, 0]):          # own class (core.U_CLS = 0)
                    self.bad += 1
                held = [int(np.argmax(v[9 * j + 3:9 * j + 9])) for j in range(10)]
                have = [0 if sim.ui[j, 1] < 0 else int(sim.it[sim.ui[j, 1], 0]) + 1 for j in range(10)]   # U_HELD = 1, I_TYPE = 0
                if held != have:
                    self.bad += 1
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--opp", default="readme", choices=("self", "readme"))
    ap.add_argument("--matches", type=int, default=4)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--time-scale", type=float, default=20.0)
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("policy", f"{args.sub}/policy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ours = Checked(load_checkpoint(mod.BlackoutPolicy, f"{args.sub}/checkpoint.pt", vector_size=96, n_channels=11))
    if args.opp == "self":
        opp = load_checkpoint(mod.BlackoutPolicy, f"{args.sub}/checkpoint.pt", vector_size=96, n_channels=11)
    else:
        torch.manual_seed(args.worker)
        net = ReadmePolicy(96, 11)
        torch.save({"policy_state": net.state_dict()}, f"/tmp/readme_{args.worker}.pt")
        opp = load_checkpoint(ReadmePolicy, f"/tmp/readme_{args.worker}.pt", vector_size=96, n_channels=11)

    kw = {"env_path": args.build}
    import inspect
    sig = inspect.signature(BlackOutEnv.__init__).parameters
    if "worker_id" in sig:
        kw["worker_id"] = args.worker
    if "time_scale" in sig:
        kw["time_scale"] = args.time_scale
    env = BlackOutEnv(**kw)
    t0 = time.time()
    series = run_series(env, ours, opp, n_matches=args.matches)
    env.close()
    pol = ours.inner._net
    print(json.dumps({
        "worker": args.worker, "opp": args.opp, "wins": series.model_a_wins, "losses": series.model_b_wins,
        "draws": series.draws,
        "matches": [{"winner": m.winner, "steps": m.episode_steps, "reward": [m.team_a_total_reward, m.team_b_total_reward]}
                    for m in series.matches],
        "mismatch": ours.bad, "calls": ours.calls, "ms_per_act": 1000 * ours.t / max(ours.calls, 1),
        "worst_ms": 1000 * ours.worst, "backend": pol._graph.backend,
        "tracker": {k: v for k, v in pol._tracker.stats.items()} if pol._tracker is not None else None,
        "wall_s": time.time() - t0}), flush=True)


if __name__ == "__main__":
    main()
