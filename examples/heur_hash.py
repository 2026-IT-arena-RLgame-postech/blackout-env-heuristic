"""Capture / check bit-exactness fingerprints of the champion heuristics (heuristics/replay_hash.py).

    PYTHONPATH=. python examples/heur_hash.py capture --out tests/data/heuristic_hashes.json --seeds 6 --steps 3000
    PYTHONPATH=. python examples/heur_hash.py check   --golden tests/data/heuristic_hashes.json --workers 4
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blackout_env.heuristics.replay_hash import BASELINES, match_hash  # noqa: E402


def _job(a):
    name, desc, seed, side, steps = a
    return f"{name}|{seed}|{side}", match_hash(desc, seed, side, steps)


def run(seeds, steps, workers, descs):
    jobs = [(n, d, 4_000_000 + s, s % 2, steps) for n, d in descs.items() for s in range(seeds)]
    with ProcessPoolExecutor(workers) as ex:
        return dict(ex.map(_job, jobs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("capture", "check"))
    ap.add_argument("--out"), ap.add_argument("--golden")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "capture":
        res = run(a.seeds, a.steps, a.workers, BASELINES)
        json.dump({"steps": a.steps, "hashes": res}, open(a.out, "w"), indent=1)
        print("captured", len(res))
    else:
        gold = json.load(open(a.golden))
        res = run(a.seeds, gold["steps"], a.workers, BASELINES)
        bad = [k for k, v in res.items() if gold["hashes"].get(k) != v]
        print(f"{len(res) - len(bad)}/{len(res)} identical; mismatches: {bad}")
        sys.exit(1 if bad else 0)
