"""Q12 (docs §46): what the search's Carrier deviations on top of V30 do -- team 6 s event deltas (chosen - base)
and the Carrier's root context.  python reports/q12_v31/scripts/car_dev.py v29,v25"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "examples")
import q11_why as q  # noqa: E402
from blackout_env.sim.mapdata import CARRIER, HUNTER  # noqa: E402

OUT = Path("reports/q12_v31")
KEYS = q.lsw.EV_KEYS  # death, kill, pickup, steal, deposit, deposit_amt, transform_h, transform_c, lost_amt
rows = []
for opp in sys.argv[1].split(","):
    S = q.load_tag(OUT, "search_b30", opp)
    C = dict(np.load(OUT / f"cf_{opp}_b30.npz"))
    f = q.lsw.root_features(S)
    dvi = q.deviation_dir(S, f)
    g = q.lsw.cf_gains(C)["vH"]
    K = int(C["K"])
    ev = C["ev"].astype(float)
    for r, i in enumerate(C["root"].astype(int)):
        if f["kind"][i] == "sam" or int(f["u_cls"][i]) != CARRIER:
            continue
        gm, t, u = int(f["game"][i]), int(f["tick"][i]), int(f["unit"][i])
        side = int(S["side"][gm])
        own, op = np.arange(5) + 5 * side, np.arange(5) + 5 * (1 - side)
        d = ev[r, K:].mean(0) - ev[r, :K].mean(0)  # [10, 9]
        p = S["pos"][t, gm, u + 5 * side].astype(float)
        oc, opos = S["cls"][t, gm, op], S["pos"][t, gm, op].astype(float)
        hd = [np.linalg.norm(opos[j] - p) for j in range(5) if oc[j] == HUNTER]
        nh_own = int((S["cls"][t, gm, own] == HUNTER).sum())
        rows.append(dict(opp=opp, g=float(g[r]), t=t / 25, bat=float(f["u_bat"][i] > 0), hd=min(hd) if hd else 99.0,
                         nho=nh_own, nhe=len(hd), dc=float(np.linalg.norm(p - q.SANCT_C)),
                         flee=dvi["d_flee"][i], sanct=dvi["d_sanct"][i], prey=dvi["d_prey"][i], cos=dvi["cos"][i],
                         **{f"o_{k}": float(d[own, j].sum()) for j, k in enumerate(KEYS)},
                         **{f"p_{k}": float(d[op, j].sum()) for j, k in enumerate(KEYS)},
                         **{f"u_{k}": float(d[u + 5 * side, j]) for j, k in enumerate(KEYS)}))
print(len(rows), "Carrier deviation roots")
cols = ["g"] + [f"o_{k}" for k in KEYS] + [f"p_{k}" for k in KEYS] + [f"u_{k}" for k in ("death", "pickup", "deposit_amt", "lost_amt")]


def show(name, m):
    sub = [r for r, x in zip(rows, m) if x]
    if not sub:
        return
    print(f"-- {name}: n {len(sub)}")
    for c in cols:
        v = np.array([r[c] for r in sub])
        print(f"   {c:16s} {v.mean():+7.2f} ± {v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0:5.2f}")


show("all", [True] * len(rows))
show("enemy Hunter < 4", [r["hd"] < 4 for r in rows])
show("enemy Hunter >= 4", [r["hd"] >= 4 for r in rows])
show("empty", [r["bat"] == 0 for r in rows])
print("-- context of all: t / dc / hd / nh own,enemy / flee / sanct / prey / cos / gain")
for r in sorted(rows, key=lambda r: -r["g"]):
    print(f"   {r['opp']} t {r['t']:5.1f} bat {r['bat']:.0f} dc {r['dc']:5.1f} hd {r['hd']:5.1f} H {r['nho']}v{r['nhe']} "
          f"flee {r['flee']:+.2f} sanct {r['sanct']:+.2f} prey {r['prey']:+.2f} cos {r['cos']:+.2f} gain {r['g']:+6.1f} "
          f"own dep {r['o_deposit_amt']:+.1f} opp dep {r['p_deposit_amt']:+.1f} own death {r['o_death']:+.1f} opp tH {r['p_transform_h']:+.1f}")
