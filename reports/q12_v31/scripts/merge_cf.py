"""Merge reports/q12_v31/cf_<opp>_b30_cp<k>.npz -> cf_<opp>_b30.npz (docs §46)."""
import sys
from pathlib import Path

import numpy as np

OUT = Path("reports/q12_v31")
for opp in sys.argv[1].split(","):
    parts = sorted(OUT.glob(f"cf_{opp}_b30_cp*.npz"))
    if not parts:
        continue
    P = [dict(np.load(p)) for p in parts]
    m = {k: (P[0][k] if P[0][k].ndim == 0 else np.concatenate([p[k] for p in P])) for k in P[0]}
    np.savez_compressed(OUT / f"cf_{opp}_b30.npz", **m)
    print(opp, [p.name for p in parts], len(m["root"]))
