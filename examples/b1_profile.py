"""B1: profile one o4f search call (blind_search_eval's o4f_v28r, base P20) over R mid-game roots."""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "examples"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roots", type=int, default=128)
    p.add_argument("--tag", default="o4f_v28r")
    p.add_argument("--base", default="p20")
    p.add_argument("--device", default="")
    p.add_argument("--packed", action="store_true")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--prof", action="store_true")
    p.add_argument("--pipeline", type=int, default=0, help="1 = PipelinedRollout (exact chunks), n > 1 = chunk_games n")
    p.add_argument("--graphs", action="store_true", help="CUDA graphs for full chunks")
    p.add_argument("--compare", action="store_true", help="also run the reference searcher; values must be equal")
    args = p.parse_args()
    import torch

    import blind_search_eval as bse
    from blackout_env.sim import SimState
    from blackout_env.sim.batch import BatchRollout, NetPolicy
    from blackout_env.sim.search import default_device

    torch.set_num_threads(1)
    dev = args.device or default_device()
    base = NetPolicy.from_checkpoint(str(REPO / bse.BASES[args.base]), device=dev)
    roots = BatchRollout(base, base).run([SimState.from_seed(4_100_000 + k) for k in range(args.roots + 16)], 750).states()
    roots = [s for s in roots if not s.over][:args.roots]
    teams = [k % 2 for k in range(len(roots))]
    seeds = list(range(len(roots)))
    ns = argparse.Namespace(v25r=bse.V25R, value=bse.VALUE, branches="alt*,samp:1.0x2", window=25, margin=3.0)
    s, _, _ = bse.build_searcher(args.tag, base, dev, ns)
    if args.packed:
        s.roll.packed = True
    ref = None
    if args.pipeline:
        from blackout_env.sim.pipeline import PipelinedRollout

        if args.compare:
            ref, _, _ = bse.build_searcher(args.tag, base, dev, ns)
        s.roll = PipelinedRollout(s.roll.pa, chunk_games=None if args.pipeline == 1 else args.pipeline, graphs=args.graphs)
    s.search(roots[:8], teams[:8], seeds[:8])
    for rep in range(args.reps):
        pr = cProfile.Profile() if args.prof and rep == args.reps - 1 else None
        t0 = time.perf_counter()
        if pr:
            pr.enable()
        out = s.search(roots, teams, seeds)
        if pr:
            pr.disable()
        wall = time.perf_counter() - t0
        tm = s.roll.timing
        print(f"rep {rep}: wall {wall:.2f}s  per root {1000 * wall / len(roots):.0f} ms  gd {tm['game_decisions']}  "
              f"render {tm['render']:.2f} policy {tm['policy']:.2f} step {tm['step']:.2f} leaf {out.timing['leaf_s']:.2f} "
              f"wait {tm.get('wait', 0):.2f} wait_base {tm.get('wait_base', 0):.2f} window {tm.get('window_s', 0):.2f} "
              f"gd/s {tm['game_decisions'] / wall:.0f}  choice {np.bincount(out.choice, minlength=8).tolist()}", flush=True)
        if ref is not None:
            t1 = time.perf_counter()
            r = ref.search(roots, teams, seeds)
            print(f"  reference wall {time.perf_counter() - t1:.2f}s  values equal {np.array_equal(r.values, out.values)}  "
                  f"choice equal {np.array_equal(r.choice, out.choice)}  max |dv| {np.abs(r.values - out.values).max():.2e}", flush=True)
        if pr:
            st = io.StringIO()
            pstats.Stats(pr, stream=st).sort_stats("tottime").print_stats(25)
            print(st.getvalue())


if __name__ == "__main__":
    main()
