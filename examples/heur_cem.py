"""Sample a CEM generation of heuristic descriptors around a base (params.sample_params) -> ';'-joined list.

    PYTHONPATH=. .venv/bin/python examples/heur_cem.py 'V32.1,commit_margin=-0.3' --n 24 --seed 1 --scale 1.0 --structural
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blackout_env.heuristics.params import sample_overrides, with_params  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--structural", action="store_true")
    p.add_argument("--groups", default="", help="comma-separated knob groups to perturb (default all)")
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    out = []
    kw = {"structural": a.structural}
    if a.groups:
        kw["groups"] = tuple(a.groups.split(","))
    while len(out) < a.n:
        ov = sample_overrides(a.base, rng, a.scale, **kw)
        if not ov:
            continue
        d = with_params(a.base, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in ov.items()})
        if d not in out:
            out.append(d)
    print(";".join(out))


if __name__ == "__main__":
    main()
