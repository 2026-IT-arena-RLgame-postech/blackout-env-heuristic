"""Q11 (docs §44): where our Hunters stand in 5-30 s (search vs plain P17): distance to the sanctuary centre, role
split, and the enemy's Hunter transforms they fail to stop.  python .../hunter_pos.py v29,v28,v25,p17"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "examples")
import q11_why as q  # noqa: E402
from blackout_env.sim import core  # noqa: E402
from blackout_env.sim.mapdata import COLLECTOR, HUNTER  # noqa: E402

OUT = Path("reports/q11_v30")
C = np.array([12.0, 12.0])


def stats(D, lo, hi):
    n = len(D["seed"])
    side = D["side"].astype(int)
    dmin, nh, guard, ecol_near = [], [], [], []
    for g in range(n):
        own = np.arange(5) + 5 * side[g]
        opp = np.arange(5) + 5 * (1 - side[g])
        for t in range(lo, min(hi, int(D["end"][g])), 5):
            cl = D["cls"][t, g]
            h = own[cl[own] == HUNTER]
            if not len(h):
                continue
            d = np.linalg.norm(D["pos"][t, g, h] - C, axis=1)
            dmin.append(d.min())
            nh.append(len(h))
            guard.append((d < 3.0).any())
            ec = opp[(cl[opp] == COLLECTOR)]
            ecol_near.append((np.linalg.norm(D["pos"][t, g, ec] - C, axis=1) < 4).sum() if len(ec) else 0)
    tr = 0
    for e in D["events"]:
        g, t, kind, a, b, c, d = map(int, e)
        if kind == core.EV_TRANSFORM and b == HUNTER and lo <= t < hi and ((a < 5) != (side[g] == 0)):
            tr += 1
    return (np.median(dmin), np.mean(guard), np.mean(nh), tr / n, np.mean(ecol_near))


for o in sys.argv[1].split(","):
    P, S = q.load_tag(OUT, "plain", o), q.load_tag(OUT, "search", o)
    for lo, hi in ((125, 250), (250, 375), (375, 500), (500, 750)):
        a, b = stats(S, lo, hi), stats(P, lo, hi)
        print(f"{o:4s} {lo // 25:2d}-{hi // 25:2d}s  own-Hunter min dist to centre med {a[0]:.2f} vs {b[0]:.2f} | "
              f"a Hunter within 3 tiles {a[1]:.2f} vs {b[1]:.2f} | own Hunters {a[2]:.2f} vs {b[2]:.2f} | "
              f"enemy H transforms/match {a[3]:.2f} vs {b[3]:.2f} | enemy Collectors < 4 of centre {a[4]:.2f} vs {b[4]:.2f}")
