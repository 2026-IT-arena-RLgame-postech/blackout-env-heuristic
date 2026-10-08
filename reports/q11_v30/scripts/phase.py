"""Q11 (docs §44): search - plain per time window -- events, Hunter counts, lead -- paired by (seed, side)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "examples")
import q11_why as q  # noqa: E402
from blackout_env.sim import core  # noqa: E402
from blackout_env.sim.mapdata import CARRIER, COLLECTOR, HUNTER  # noqa: E402

OUT = Path("reports/q11_v30")
WIN = [(5, 10), (10, 15), (15, 20), (20, 30), (30, 60), (60, 150)]
KEYS = ["dep", "odep", "d_col", "d_car", "d_hun", "od_col", "od_car", "od_hun", "lost", "olost", "tr_h", "otr_h",
        "tr_c", "otr_c", "steal", "osteal", "pick", "opick"]


def facts(D):
    n = len(D["seed"])
    side = D["side"].astype(int)
    F = {k: np.zeros((n, len(WIN))) for k in KEYS}
    for e in D["events"]:
        g, t, kind, a, b, c, d = map(int, e)
        w = [i for i, (lo, hi) in enumerate(WIN) if lo * 25 <= t < hi * 25]
        if not w:
            continue
        w = w[0]
        mine = lambda u: (u < 5) == (side[g] == 0)  # noqa: E731
        if kind == core.EV_KILL:
            cl = {COLLECTOR: "col", CARRIER: "car", HUNTER: "hun"}[c]
            F[("d_" if mine(b) else "od_") + cl][g, w] += 1
        elif kind == core.EV_DEPOSIT and b == 0:
            F["dep" if mine(a) else "odep"][g, w] += c
        elif kind == core.EV_LOST and b == 0:
            F["lost" if mine(a) else "olost"][g, w] += c
        elif kind == core.EV_TRANSFORM:
            if b == HUNTER:
                F["tr_h" if mine(a) else "otr_h"][g, w] += 1
            elif b == CARRIER:
                F["tr_c" if mine(a) else "otr_c"][g, w] += 1
        elif kind == core.EV_STEAL:
            F["steal" if mine(a) else "osteal"][g, w] += 1
        elif kind == core.EV_PICKUP:
            F["pick" if mine(a) else "opick"][g, w] += 1
    for s in (5, 10, 15, 20, 30):
        t = np.minimum(s * 25, D["end"])
        c = D["cls"][t, np.arange(n)]
        own = np.stack([c[i, 5 * side[i]:5 * side[i] + 5] for i in range(n)])
        opp = np.stack([c[i, 5 * (1 - side[i]):5 * (1 - side[i]) + 5] for i in range(n)])
        for nm, arr in (("h", HUNTER), ("car", CARRIER)):
            F[f"own_{nm}@{s}"] = (own == arr).sum(1)[:, None].astype(float)
            F[f"opp_{nm}@{s}"] = (opp == arr).sum(1)[:, None].astype(float)
        sc = D["score"][t, np.arange(n)]
        F[f"lead@{s}"] = np.where(side == 0, sc[:, 0] - sc[:, 1], sc[:, 1] - sc[:, 0])[:, None].astype(float)
    return F


def main():
    opps = sys.argv[1].split(",")
    tag = sys.argv[2] if len(sys.argv) > 2 else "search"
    for o in opps:
        P, S = q.load_tag(OUT, "plain", o), q.load_tag(OUT, tag, o)
        n = min(len(P["seed"]), len(S["seed"]))
        assert (P["seed"][:n] == S["seed"][:n]).all()
        fp, fs = facts(P), facts(S)
        seeds = S["seed"][:n]
        print(f"== {tag} - plain vs {o} ({n} matches)")
        for k in fp:
            a, b = fs[k][:n], fp[k][:n]
            if a.shape[1] == 1:
                m, e, _ = q.lsw.paired(seeds, (a - b)[:, 0])
                print(f"   {k:12s} search {a.mean():6.2f} plain {b.mean():6.2f}  d {m:+6.2f} ± {e:.2f}")
            else:
                cells = []
                for w in range(len(WIN)):
                    m, e, _ = q.lsw.paired(seeds, a[:, w] - b[:, w])
                    cells.append(f"{b[:, w].mean():5.2f}{m:+6.2f}±{e:4.2f}")
                print(f"   {k:7s} " + "  ".join(cells))
        print("   (window cells: plain mean, search - plain; windows " + ", ".join(f"{a}-{b}s" for a, b in WIN) + ")")


if __name__ == "__main__":
    main()
