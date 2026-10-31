"""Focused search: CEM over one knob group around a base, distributed with bp_loop.play, then confirm the top-k.

    nohup .venv/bin/python scripts/bp_focus.py NAME BASE_DESC --groups race --n 24 --opps 'd1;d2' &
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bp_loop as L
from blackout_env.heuristics.params import sample_overrides, with_params

ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("base")
ap.add_argument("--groups", default="race"); ap.add_argument("--n", type=int, default=24)
ap.add_argument("--scale", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--opps", required=True); ap.add_argument("--seeds", type=int, default=32)
ap.add_argument("--confirm", type=int, default=96); ap.add_argument("--topk", type=int, default=4)
ap.add_argument("--extra", default="", help="';'-joined hand-made candidates added to the CEM ones")
a = ap.parse_args()
opps = a.opps.split(";")
cf = L.OUT / f"{a.name}_cands.json"
if cf.exists():
    cands = json.loads(cf.read_text())
else:
    rng = np.random.default_rng(a.seed); cands = []
    while len(cands) < a.n:
        ov = sample_overrides(a.base, rng, a.scale, groups=tuple(a.groups.split(",")), structural=True, exclude=L.PARKED)
        if ov:
            d = with_params(a.base, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in ov.items()})
            if d not in cands: cands.append(d)
    cands += [c for c in a.extra.split(";") if c]
    cf.write_text(json.dumps(cands))
rows = L.play(f"{a.name}_s", [a.base] + cands, opps, a.seeds, 50_000_000 + 1000 * a.seed)
t = L.table(rows)
sc = sorted(((L.paired(t, c, a.base, opps), c) for c in cands if c in t), key=lambda x: -np.nan_to_num(x[0][0], nan=-1e9))
L.log(f"{a.name} screen: " + "; ".join(f"{s[0]:+.1f}±{s[1]:.1f}" for s, _ in sc[:8]))
top = [c for _, c in sc[:a.topk]]
rows = L.play(f"{a.name}_c", [a.base] + top, opps, a.confirm, 60_000_000 + 1000 * a.seed)
t = L.table(rows)
conf = sorted(((L.paired(t, c, a.base, opps), c) for c in top), key=lambda x: -np.nan_to_num(x[0][0], nan=-1e9))
L.log(f"{a.name} confirm: " + "; ".join(f"{s[0]:+.1f}±{s[1]:.1f}" for s, _ in conf))
(L.OUT / f"{a.name}_result.json").write_text(json.dumps([{"d": s[0], "se": s[1], "desc": c} for s, c in conf], indent=1))
