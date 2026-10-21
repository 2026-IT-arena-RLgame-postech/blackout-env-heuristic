"""Sample parameterised variants of a champion heuristic and measure strength + behavioural diversity.

    # print 8 sampled descriptors (seeded)
    PYTHONPATH=. python examples/heur_variants.py sample V32.1 --n 8 --seed 0 --scale 1.0
    # play them (both sides) vs a reference opponent in the Python sim; strength + disagreement with the default
    PYTHONPATH=. python examples/heur_variants.py run V32.1 --n 8 --seeds 16 --ref V31 --workers 4 --out reports/heur_variants/demo.json
    # which knobs change behaviour at all (lo / hi value vs the base, action hash + disagreement)
    PYTHONPATH=. python examples/heur_variants.py sensitivity V32.1 --seeds 2 --workers 4

Strength = win rate / mean score margin of the variant vs ``--ref`` over ``--seeds`` maps, each played on both sides.
Diversity = *action disagreement*: a shadow instance of the base heuristic is fed the very same observations as the
variant (so both are in the same state), and a unit-decision counts as a disagreement when the two moves differ in
heading by more than 22.5 degrees or one stands and the other moves.  ``idle`` = share of unit-decisions with no move
(a degenerate variant freezes).  Everything is Python-sim, CPU only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blackout_env.heuristics.params import (KNOBS, base_values, resolve_descriptor, sample_overrides, split_descriptor,  # noqa: E402
                                            validate, with_params)

MAX_STEPS = 12000
STAND = 1e-5


def play(variant: str, base: str, ref: str, seed: int, side: int, steps: int = MAX_STEPS) -> dict:
    """``variant`` (team ``side``) vs ``ref`` on map ``seed``, with a shadow ``base`` fed the same observations."""
    from blackout_env.sim.state import SimState, obs_from_raw
    from blackout_env.train.distill.opponents import make_opponent

    sim = SimState.from_seed(seed)
    me, opp, shadow = make_opponent(variant, seed), make_opponent(ref, seed + 1), make_opponent(base, seed)
    dis = idle = units = ndiff = 0
    first = -1
    n = 0
    for n in range(1, steps + 1):
        obs = obs_from_raw(sim.raw_state(), sim.raw_map())
        mv = np.zeros((10, 2), np.float32)
        mine = {f"unit_{i}": obs[f"unit_{i}"] for i in range(5 * side, 5 * side + 5)}
        theirs = {f"unit_{i}": obs[f"unit_{i}"] for i in range(5 * (1 - side), 5 * (1 - side) + 5)}
        a_me, a_sh = me.act(mine), shadow.act(mine)
        for team, acts in ((side, a_me), (1 - side, opp.act(theirs))):
            for name, act in acts.items():
                mv[int(name.split("_")[1])] = np.clip(np.asarray(act, np.float32), -1, 1)
        for name in a_me:
            x = np.clip(np.asarray(a_me[name], np.float32), -1, 1)
            y = np.clip(np.asarray(a_sh.get(name, (0, 0)), np.float32), -1, 1)
            nx, ny = float(np.hypot(*x)), float(np.hypot(*y))
            units += 1
            if not np.array_equal(x, y):
                ndiff += 1
                first = n if first < 0 else first
            idle += nx < STAND
            if (nx < STAND) != (ny < STAND):
                dis += 1
            elif nx >= STAND and float(x @ y) / (nx * ny) < np.cos(np.radians(22.5)):
                dis += 1
        sim.step(mv)
        if sim.over:
            break
    a, b = sim.scores
    own, other = (a, b) if side == 0 else (b, a)
    return {"margin": int(own - other), "win": own > other, "steps": n, "dis": dis / max(units, 1), "ndiff": ndiff, "first": first,
            "idle": idle / max(units, 1)}


def _job(args):
    return play(*args)


def run_variants(variants: dict[str, str], base: str, ref: str, seeds: int, workers: int, seed0: int = 5_000_000):
    jobs, keys = [], []
    for name, desc in variants.items():
        for s in range(seeds):
            for side in (0, 1):
                jobs.append((desc, base, ref, seed0 + s, side))
                keys.append(name)
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_job, jobs, chunksize=1))
    out = {}
    for name in variants:
        rs = [r for k, r in zip(keys, res) if k == name]
        out[name] = {"desc": variants[name], "games": len(rs),
                     "win": float(np.mean([r["win"] for r in rs])), "margin": float(np.mean([r["margin"] for r in rs])),
                     "margin_se": float(np.std([r["margin"] for r in rs]) / np.sqrt(len(rs))),
                     "dis": float(np.mean([r["dis"] for r in rs])), "idle": float(np.mean([r["idle"] for r in rs])),
                     "fired": float(np.mean([r["ndiff"] > 0 for r in rs])),
                     "ndiff": float(np.mean([r["ndiff"] for r in rs])),
                     "steps": float(np.mean([r["steps"] for r in rs]))}
    return out


def cmd_sample(a):
    rng = np.random.default_rng(a.seed)
    for _ in range(a.n):
        d = with_params(a.base, **sample_overrides(a.base, rng, a.scale, structural=a.structural))
        validate(d)
        print(d)


def cmd_run(a):
    rng = np.random.default_rng(a.seed)
    base = resolve_descriptor(a.base)
    variants = {"base": base}
    for i in range(a.n):
        variants[f"v{i}"] = with_params(a.base, **sample_overrides(a.base, rng, a.scale, structural=a.structural))
    t = time.time()
    out = run_variants(variants, base, a.ref, a.seeds, a.workers)
    print(f"{'name':5} {'win':>5} {'margin':>14} {'dis':>6} {'idle':>6}  overrides")
    _, bv = base_values(a.base)
    for name, r in out.items():
        diff = "" if name == "base" else ",".join(
            f"{k}={v}" for k, v in sorted(split_descriptor(r['desc'])[1].items())
            if k in KNOBS and bv.get(k) != v)
        print(f"{name:5} {r['win']:5.2f} {r['margin']:+7.1f}+-{r['margin_se']:4.1f} {r['dis']:6.3f} {r['idle']:6.3f}  {diff}")
    print(f"({time.time() - t:.0f}s, ref={a.ref}, {a.seeds} maps x 2 sides)")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"base": a.base, "ref": a.ref, "scale": a.scale, "seed": a.seed, "results": out}, open(a.out, "w"), indent=1)


def cmd_sensitivity(a):
    base = resolve_descriptor(a.base)
    _, values = base_values(a.base)
    variants = {}
    for name, v in values.items():
        k = KNOBS[name]
        if k.kind == "bool":
            variants[f"{name}={not v}"] = with_params(a.base, **{name: not v})
            continue
        for end in (k.lo, k.hi):
            if end != v and not (name == "late_hunter_quota" and end < 0):
                variants[f"{name}={k.clip(end)}"] = with_params(a.base, **{name: k.clip(end)})
    per_ref = {}
    for ref in a.ref.split(";"):
        per_ref[ref] = run_variants(variants, base, ref, a.seeds, a.workers)
    print(f"{'variant':30}" + "".join(f" | {r[:14]:>14} win  margin  fired  dis" for r in per_ref))
    inert = []
    for n in variants:
        cells = [per_ref[r][n] for r in per_ref]
        print(f"{n:30}" + "".join(f" | {'':>14} {c['win']:4.2f} {c['margin']:+7.1f} {c['fired']:5.2f} {c['dis']:.4f}" for c in cells))
        if all(c["fired"] == 0.0 for c in cells):
            inert.append(n)
    print("never fired (no action differed in any game vs any ref):", inert)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"base": a.base, "refs": list(per_ref), "results": per_ref, "inert": inert}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("sample", "run", "sensitivity"))
    ap.add_argument("base", nargs="?", default="V32.1")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--structural", action="store_true", help="also flip rule switches")
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--ref", default="V31", help="reference opponent descriptor(s); sensitivity takes several separated by ;")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out")
    args = ap.parse_args()
    {"sample": cmd_sample, "run": cmd_run, "sensitivity": cmd_sensitivity}[args.cmd](args)
