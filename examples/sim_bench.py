"""Simulator speed (S3): microseconds per decision step, alone and with policies in the loop, and
the implied rollouts per second for a 2 s (50-step) rollout from a forked state.

    PYTHONPATH=. .venv/bin/python -m examples.sim_bench --trace reports/sim/traces/trace_016.npz
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from blackout_env.sim import SimState
from blackout_env.sim.trace import Trace


def _timeit(fn, n):
    fn()  # warm-up / compile
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1e6


def bench(trace_path: str, t0: int, student: str | None, n: int = 2000) -> dict:
    tr = Trace.load(trace_path)
    base = SimState.from_raw(tr.raw_state[t0], tr.raw_map[t0])
    base.reseed(1)
    moves = tr.moves[t0:t0 + 50].copy()
    out = {}

    sim = base.copy()
    k = [0]

    def step_only():
        if sim.over:
            sim.load(base)
        sim.step(moves[k[0] % 50])
        k[0] += 1

    out["step_us"] = _timeit(step_only, n)
    out["run_50_steps_us"] = _timeit(lambda: (sim.load(base), sim.run(moves)), n // 10)
    out["copy_us"] = _timeit(lambda: base.copy(), n)
    out["load_us"] = _timeit(lambda: sim.load(base), n)
    out["raw_obs_us"] = _timeit(lambda: (sim.raw_state(), sim.raw_map()), n)
    out["obs_dict_us"] = _timeit(lambda: sim.obs(), n // 4)

    from blackout_env.env.constants import team_a_agents, team_b_agents
    from blackout_env.train.distill.opponents import make_opponent

    names_a, names_b = team_a_agents(), team_b_agents()

    def loop_with(pol_a, pol_b, steps):
        sim.load(base)
        for p in (pol_a, pol_b):
            if hasattr(p, "reset"):
                p.reset()
        t = time.perf_counter()
        done = 0
        for _ in range(steps):
            if sim.over:
                sim.load(base)
            obs = sim.obs()
            acts = pol_a.act({a: obs[a] for a in names_a})
            acts.update(pol_b.act({a: obs[a] for a in names_b}))
            mv = np.zeros((10, 2), np.float32)
            for name, a in acts.items():
                mv[int(name.split("_")[1])] = np.clip(a, -1, 1)
            sim.step(mv)
            done += 1
        return (time.perf_counter() - t) / done * 1e6

    for label, desc in (("v21_anyangle", "strategic_v21_anyangle"), ("v17", "strategic_v17"), ("random", "random")):
        pa, pb = make_opponent(desc, 1), make_opponent(desc, 2)
        loop_with(pa, pb, 20)  # warm caches
        out[f"step_with_{label}_both_us"] = loop_with(pa, pb, 300)
    if student:
        import torch

        from blackout_env.train.distill.inference import load_net, make_student

        torch.set_num_threads(1)
        net = load_net(torch.load(student, map_location="cpu", weights_only=True)["policy_state"])
        pa, pb = make_student(net, "fixed_tau"), make_student(net, "fixed_tau")
        loop_with(pa, pb, 10)
        out["step_with_student_both_us"] = loop_with(pa, pb, 100)
        h = make_opponent("strategic_v21_anyangle", 1)
        loop_with(pa, h, 10)
        out["step_with_student_vs_v21_us"] = loop_with(pa, h, 100)
    steps = 50
    for key in [k for k in out if k.startswith("step")]:
        out[key.replace("_us", "_rollouts2s_per_s")] = 1e6 / (out[key] * steps + out["load_us"])
    out["rollouts2s_per_s_sim_only_run"] = 1e6 / out["run_50_steps_us"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace", default="reports/sim/traces/trace_016.npz")
    parser.add_argument("--t0", type=int, default=2500)
    parser.add_argument("--student", default="checkpoints/distill/frozen/p5a_v22.pt")
    parser.add_argument("--out", default="reports/sim/bench.json")
    args = parser.parse_args()
    res = bench(args.trace, args.t0, args.student or None)
    print(json.dumps(res, indent=1))
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
