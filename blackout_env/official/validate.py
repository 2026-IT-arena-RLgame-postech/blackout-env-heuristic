"""Tracker accuracy on simulated matches: per-tick truth vs reconstruction from official obs.

    PYTHONPATH=. python -m blackout_env.official.validate --seeds 0-3 --decisions 3000
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from numba import njit

from ..sim import core
from ..sim.state import SimState
from .render import official_map, official_vector
from .tracker import MultiTracker, OfficialTracker

F32 = np.float32


@njit(cache=True)
def true_tick(S, M, moves, order):
    """One Unity tick of the official build cadence (every unit moves every tick)."""
    g = S[5]
    g[core.G_NEV] = 0
    for j in range(order.shape[0]):
        u = order[j]
        if g[core.G_OVER]:
            return
        core.move_unit(S, M, u, moves[u, 0], moves[u, 1])
    if g[core.G_OVER] == 0:
        core.world_update(S, M)
    g[core.G_TICK] += 1


def compare(tr: OfficialTracker, sim: SimState, acc: dict):
    a, b = tr.raw_state(), sim.raw_state()
    ma, mb = tr.raw_map().astype(np.int64), sim.raw_map().astype(np.int64)
    blk_a, blk_b = a[:40].reshape(10, 4), b[:40].reshape(10, 4)
    held_t = (blk_a[:, 2] != 0) == (blk_b[:, 2] != 0)
    acc["n"] += 1
    acc["pos"] += np.all(blk_a[:, :2] == blk_b[:, :2])
    acc["cls"] += np.all(blk_a[:, 3] == blk_b[:, 3])
    acc["held_type"] += np.all(held_t & (np.sign(blk_a[:, 2]) == np.sign(blk_b[:, 2])))
    acc["held_exact"] += np.all(blk_a[:, 2] == blk_b[:, 2])
    acc["scalars"] += np.all(a[40:] == b[40:])
    na, nb = (ma >> 3) & 15, (mb >> 3) & 15
    acc["map_base"] += np.all((ma & 7) == (mb & 7))
    acc["map_items"] += np.all((ma >> 7) == (mb >> 7))
    acc["bat_presence"] += np.all((na > 0) == (nb > 0))
    acc["bat_exact"] += np.all(na == nb)
    acc["bat_l1"] += int(np.abs(na - nb).sum())
    acc["all_exact"] += np.array_equal(a, b) and np.array_equal(ma, mb)


class OracleFinder:
    def __init__(self, state):
        self.state = state

    def __call__(self, chosen, tiles, prev=None):
        return self.state, "oracle"


def run(seed: int, decisions: int, desc_a: str, desc_b: str, finder: str, observer: int = 0, tr=None,
        gen_state=None, blind=False):
    """One match.  ``gen_state``: generation RNG state (default InitState(seed)); ``tr``: a tracker
    carried over from the previous match (chained unseeded runs)."""
    from ..sim.unity_random import init_state
    from ..train.distill.opponents import make_opponent
    from .seed_search import RngFinder

    gen_state = init_state(seed) if gen_state is None else gen_state
    sim = SimState.from_rng_state(gen_state)
    pa, pb = make_opponent(desc_a, 2 * seed), make_opponent(desc_b, 2 * seed + 1)
    if tr is None:
        f = {"none": None, "oracle": OracleFinder(gen_state), "search": RngFinder(full_scan=False)}[finder]
        tr = MultiTracker(f) if blind else OfficialTracker(rng_finder=f)
    see = (lambda s_: None) if blind else official_map
    acc = {k: 0 for k in ("n", "pos", "cls", "held_type", "held_exact", "scalars", "map_base", "map_items",
                          "bat_presence", "bat_exact", "bat_l1", "all_exact")}
    tr.observe(official_vector(sim, observer), see(sim))
    compare(tr, sim, acc)
    moves = np.zeros((10, 2), F32)
    for _ in range(decisions):
        if sim.over:
            break
        obs = sim.obs()
        for pol, units in ((pa, range(5)), (pb, range(5, 10))):
            acts = pol.act({f"unit_{u}": obs[f"unit_{u}"] for u in units})
            for name, act in acts.items():
                moves[int(name.split("_")[1])] = act
        for _ in range(2):
            true_tick(sim.S, sim.M, moves, sim.order)
            if sim.over:
                break
            tr.observe(official_vector(sim, observer), see(sim))
            compare(tr, sim, acc)
    return acc, tr.stats, sim.scores, tr, sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-3")
    ap.add_argument("--decisions", type=int, default=3000)
    ap.add_argument("--a", default="strategic_v25")
    ap.add_argument("--b", default="strategic_v22")
    ap.add_argument("--finder", default="oracle", choices=("none", "oracle", "search"))
    ap.add_argument("--chain", action="store_true",
                    help="official unseeded run: matches continue one RNG stream from InitState(0)+109")
    ap.add_argument("--blind", action="store_true", help="no map (headless official build): MultiTracker")
    ap.add_argument("--boundary-gens", default="1",
                    help="wasted generations between chained matches, comma list cycled (official run_match: 1)")
    args = ap.parse_args()
    lo, _, hi = args.seeds.partition("-")
    seeds = range(int(lo), int(hi or lo) + 1)
    tot, tr, gen = None, None, None
    if args.chain:
        from .seed_search import advance
        from ..sim.unity_random import init_state
        gen = advance(init_state(0), 109)
    for s in seeds:
        t0 = time.time()
        acc, stats, scores, tr_, sim = run(s, args.decisions, args.a, args.b, args.finder,
                                           observer=0 if s % 2 == 0 else 7, tr=tr, gen_state=gen, blind=args.blind)
        if args.chain:
            from ..sim.unity_random import generate_state
            tr, gen = tr_, sim.rng.copy()
            gens = [int(x) for x in args.boundary_gens.split(",")]
            for _ in range(gens[(s - seeds[0]) % len(gens)]):
                gen = generate_state(gen)[2]
        n = acc["n"]
        print(f"seed {s} ticks {n} scores {scores} {time.time() - t0:.0f}s  " +
              " ".join(f"{k}={acc[k] / n:.4f}" for k in acc if k not in ("n", "bat_l1")) +
              f" bat_l1/tick={acc['bat_l1'] / n:.2f}", flush=True)
        print("   stats", stats, flush=True)
        tot = acc if tot is None else {k: tot[k] + acc[k] for k in acc}
    n = tot["n"]
    print("TOTAL " + " ".join(f"{k}={tot[k] / n:.4f}" for k in tot if k not in ("n", "bat_l1")) +
          f" bat_l1/tick={tot['bat_l1'] / n:.2f}")


if __name__ == "__main__":
    main()
