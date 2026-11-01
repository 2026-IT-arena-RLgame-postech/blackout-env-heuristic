"""Hand-made candidates vs a pool, paired against the base: python scripts/bp_cand.py NAME BASE 'extra1;extra2' 'opp1;opp2' SEEDS [SEED0]
Each extra is a knob string appended to BASE (``k=v,k=v``)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bp_loop as L
from blackout_env.heuristics.params import resolve_descriptor
name, base, extras, opps, seeds = sys.argv[1], sys.argv[2], sys.argv[3].split(";"), sys.argv[4].split(";"), int(sys.argv[5])
seed0 = int(sys.argv[6]) if len(sys.argv) > 6 else 80_000_000
b = resolve_descriptor(base); od = [resolve_descriptor(o) for o in opps]
cands = {e: b + "," + e for e in extras}
rows = L.play(name, [b] + list(cands.values()), od, seeds, seed0)
t = L.table(rows)
out = [f"{name} base={base} seeds={seeds}x2 seed0={seed0}"]
for e, d in cands.items():
    tot = L.paired(t, d, b, od)
    per = " ".join(f"{o}:{L.paired(t, d, b, [x])[0]:+.1f}" for o, x in zip(opps, od))
    out.append(f"{e:60s} paired {tot[0]:+.1f}±{tot[1]:.1f} raw {tot[2]:+.1f} | {per}")
L.log("\n".join(out))
