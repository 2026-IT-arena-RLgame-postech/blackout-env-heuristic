"""Merge the cf part files reports/q11_v30/cf_<opp><suffix>p<k>.npz -> cf_<opp>.npz (docs §44)."""
import sys
from pathlib import Path

import numpy as np

OUT = Path("reports/q11_v30")
for opp in sys.argv[1].split(","):
    parts = sorted(OUT.glob(f"cf_{opp}_c*p*.npz"))
    if not parts:
        continue
    P = [dict(np.load(p)) for p in parts]
    m = {k: (P[0][k] if P[0][k].ndim == 0 else np.concatenate([p[k] for p in P])) for k in P[0]}
    np.savez_compressed(OUT / f"cf_{opp}.npz", **m)
    print(opp, [p.name for p in parts], len(m["root"]))
