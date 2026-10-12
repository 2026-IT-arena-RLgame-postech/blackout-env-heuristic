"""End-to-end check of the built official submission without Unity.

The submission (``policy.py`` + ``checkpoint.pt`` from examples/build_official_submission.py) is
loaded with the *official* ``load_checkpoint`` (ist-tech-AI-games/blackout-env, imported from
``--official-env`` under another package name) and driven exactly like ``run_match`` would: per
tick, each team's 5 agents' official preprocessed obs (``official_preprocessed``) -> ``act`` ->
one official-cadence truth tick (``validate.true_tick``).  Matches chain one unseeded RNG stream
(InitState(0) + startup, then end state + 1 generation), as the official runner's resets do.

The submission plays one side, a fork policy (``--opp``) on the true fork obs the other.  At every
decision tick its moves are compared with the P24c reference (``make_student`` on the *true* fork
obs): any difference means the tracker or the graph went wrong.

    PYTHONPATH=. .venv/bin/python examples/official_submission_check.py --dir dist/official_submission --matches 4
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np

from blackout_env.official.render import official_map, official_preprocessed
from blackout_env.official.seed_search import advance
from blackout_env.official.validate import true_tick
from blackout_env.sim.state import SimState
from blackout_env.sim.unity_random import generate_state, init_state
from blackout_env.train.distill.inference import load_checkpoint_net, make_student
from blackout_env.train.distill.opponents import make_opponent

F32 = np.float32


def official_loader(official_env: str):
    """The official package's ``load_checkpoint`` (its own name clashes with ours: load it as ``official_bo``)."""
    pkg = types.ModuleType("official_bo")
    pkg.__path__ = [str(Path(official_env) / "blackout_env")]
    sys.modules["official_bo"] = pkg
    model = types.ModuleType("official_bo.model")
    model.__path__ = [str(Path(official_env) / "blackout_env" / "model")]
    sys.modules["official_bo.model"] = model
    spec = importlib.util.spec_from_file_location("official_bo.model.loader", model.__path__[0] + "/loader.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.load_checkpoint


def load_policy_class(policy_path: str, class_name: str = "BlackoutPolicy"):
    """README's 'Different architectures per team' loader."""
    spec = importlib.util.spec_from_file_location("policy", policy_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="dist/official_submission")
    ap.add_argument("--official-env", required=True, help="clone of ist-tech-AI-games/blackout-env")
    ap.add_argument("--reference", default="checkpoints/distill/frozen/p24c_v321.pt")
    ap.add_argument("--opp", default="strategic_v30")
    ap.add_argument("--matches", type=int, default=4)
    ap.add_argument("--max-ticks", type=int, default=21000)
    ap.add_argument("--map", action="store_true", help="show the real map (default: the headless build's constant one)")
    ap.add_argument("--two-files", action="store_true", help="load from a copy holding only policy.py + checkpoint.pt")
    args = ap.parse_args()

    src = Path(args.dir)
    if args.two_files:
        tmp = Path(tempfile.mkdtemp())
        for f in ("policy.py", "checkpoint.pt"):
            shutil.copy(src / f, tmp / f)
        src = tmp
    load_checkpoint = official_loader(args.official_env)
    Policy = load_policy_class(str(src / "policy.py"))
    model = load_checkpoint(Policy, str(src / "checkpoint.pt"), state_dict_key="policy_state", device="cpu",
                            vector_size=96, n_channels=11)
    ref = make_student(load_checkpoint_net(args.reference), "iqn")

    gen = advance(init_state(0), 109)   # Academy InitState(0) + the startup generation
    total = {"decisions": 0, "diff": 0, "ticks": 0}
    for m in range(args.matches):
        sim = SimState.from_rng_state(gen)
        me = m % 2                         # the submission's side; sides swap like run_series
        opp = make_opponent(args.opp, m)
        mine = [f"unit_{5 * me + k}" for k in range(5)]
        theirs = [f"unit_{5 * (1 - me) + k}" for k in range(5)]
        moves = np.zeros((10, 2), F32)
        diff, decisions, t_act, worst = 0, 0, 0.0, 0.0
        roots, root_diff, root_err = 0, 0, 0.0
        for tick in range(args.max_ticks):
            if sim.over:
                break
            pre = {}
            for u in range(5 * me, 5 * me + 5):
                o = official_preprocessed(sim, u)
                if not args.map:   # the headless official build's constant map
                    o["graphic"] = np.zeros_like(o["graphic"])
                    o["graphic"][..., 0] = 1.0
                pre[f"unit_{u}"] = o
            t0 = time.perf_counter()
            sp = model._net._search
            n_search = sp.stats["searches"] if sp is not None else 0
            acts = model.act(pre)
            if sp is not None and sp.stats["searches"] > n_search:   # same search from the true state
                _, reseed, got = sp.last
                ref_out = sp.searcher.search([sim.copy()], [me], [reseed])
                roots += 1
                root_diff += int(int(ref_out.choice[0]) != int(got.choice[0]))
                root_err = max(root_err, float(np.abs(ref_out.values - got.values).max()))
            dt = time.perf_counter() - t0
            t_act += dt
            worst = max(worst, dt) if tick > 0 else worst
            for n, a in acts.items():
                moves[int(n[5:])] = a
            if tick % 2 == 0:
                fork = sim.obs()
                want = ref.act({n: fork[n] for n in mine})
                got = np.stack([acts[n] for n in mine])
                exp = np.stack([want[n] for n in mine])
                decisions += 1
                diff += int((np.abs(got - exp).max(axis=1) > 1e-3).any())
                for n, a in opp.act({n: fork[n] for n in theirs}).items():
                    moves[int(n[5:])] = a
            true_tick(sim.S, sim.M, moves, sim.order)
        total["decisions"] += decisions
        total["diff"] += diff
        total["ticks"] += tick
        total["margin_sum"] = total.get("margin_sum", 0) + int(sim.scores[me] - sim.scores[1 - me])
        st = model._net._tracker.stats
        print(f"match {m} side {'AB'[me]} ticks {tick} scores {sim.scores} decisions {decisions} differing {diff} "
              f"act {1000 * t_act / max(tick, 1):.2f} ms/tick (worst {1000 * worst:.0f} ms) rng {model._net._tracker.seed} "
              f"backend {model._net._graph.backend} stats {st}", flush=True)
        if model._net._search is not None:
            ss = model._net._search.stats
            print(f"   search: roots {roots} choice differs from true-root search {root_diff} max value err {root_err:.2e} "
                  f"{ss['search_s'] / max(ss['searches'], 1):.2f} s/root (worst {ss['worst_s']:.1f}) chosen {ss['chosen'].tolist()}",
                  flush=True)
        gen = generate_state(sim.rng.copy())[2]   # in-Unity restart + env.reset(): next match = end + 1 generation
    print("TOTAL", total, f"agreement {1 - total['diff'] / max(total['decisions'], 1):.6f}")


if __name__ == "__main__":
    main()
