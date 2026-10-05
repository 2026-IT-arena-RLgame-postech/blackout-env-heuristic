"""Q11 (docs §44): per deviating-unit class, the deviating unit's own 6 s event change (chosen - base arm, mean over
reseeds) and its context at the root, binned, with the realised gain.  python .../cf_units.py v29,v28,v25 [Car]"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "examples")
import q11_why as q  # noqa: E402
from blackout_env.sim.mapdata import CARRIER, COLLECTOR, HUNTER  # noqa: E402

OUT = Path("reports/q11_v30")
EVK = q.lsw.EV_KEYS  # death, kill, pickup, steal, deposit, deposit_amt, transform_h, transform_c, lost_amt
NAME = {COLLECTOR: "Col", HUNTER: "Hun", CARRIER: "Car"}


def main():
    opps = sys.argv[1].split(",")
    want = sys.argv[2] if len(sys.argv) > 2 else "Car"
    allrows = []
    for o in opps:
        S = q.load_tag(OUT, "search", o)
        C = dict(np.load(OUT / f"cf_{o}.npz"))
        f = q.lsw.root_features(S)
        dvi = q.deviation_dir(S, f)
        g = q.lsw.cf_gains(C)["vH"]
        K = int(C["K"])
        ev = C["ev"].astype(float)
        for r, i in enumerate(C["root"].astype(int)):
            u = int(f["unit"][i])
            if u < 0 or NAME[int(f["u_cls"][i])] != want:
                continue
            gm, t = int(f["game"][i]), int(f["tick"][i])
            side = int(S["side"][gm])
            gu = u + 5 * side
            d = ev[r, K:, gu].mean(0) - ev[r, :K, gu].mean(0)
            P, Cl = S["pos"][t, gm].astype(float), S["cls"][t, gm]
            opp = np.arange(5) + 5 * (1 - side)
            dist = np.linalg.norm(P[opp] - P[gu], axis=1)
            dcol = dist[Cl[opp] == COLLECTOR].min() if (Cl[opp] == COLLECTOR).any() else np.inf
            dhun = dist[Cl[opp] == HUNTER].min() if (Cl[opp] == HUNTER).any() else np.inf
            allrows.append(dict(opp=o, g=g[r], t=t / 25, bat=int(f["u_bat"][i]), dcol=dcol, dhun=dhun,
                                dsan=float(np.linalg.norm(P[gu] - 12.0)), ds=dvi["d_sanct"][i], df=dvi["d_flee"][i],
                                cos=dvi["cos"][i], **{k: d[j] for j, k in enumerate(EVK)}))
    R = allrows
    print(f"{want}: {len(R)} cf roots; columns: n, realised gain, d(death, pickup, steal, deposit_amt, lost_amt) of the unit")

    def show(nm, key):
        by = defaultdict(list)
        for r in R:
            by[key(r)].append(r)
        print(f"-- {nm}")
        for k in sorted(by, key=str):
            v = by[k]
            gg = [x["g"] for x in v]
            per = {o: np.mean([x["g"] for x in v if x["opp"] == o]) for o in opps if any(x["opp"] == o for x in v)}
            print(f"   {str(k):22s} n {len(v):4d} gain {np.mean(gg):+6.2f}±{q.lsw.se(gg):4.2f}  " +
                  " ".join(f"{e} {np.mean([x[e] for x in v]):+5.2f}" for e in ("death", "pickup", "steal", "deposit_amt", "lost_amt"))
                  + "   " + " ".join(f"{o} {m:+5.1f}" for o, m in per.items()))
    show("cargo", lambda r: "bat" if r["bat"] > 0 else "empty")
    show("enemy Collector distance", lambda r: "<1.5" if r["dcol"] < 1.5 else "<3" if r["dcol"] < 3 else "<6" if r["dcol"] < 6 else ">=6")
    show("enemy Hunter distance", lambda r: "<2" if r["dhun"] < 2 else "<4" if r["dhun"] < 4 else ">=4/none")
    show("dist to centre", lambda r: "<3" if r["dsan"] < 3 else "<6" if r["dsan"] < 6 else "<9" if r["dsan"] < 9 else ">=9")
    show("time", lambda r: "5-10" if r["t"] < 10 else "10-20" if r["t"] < 20 else "20-40" if r["t"] < 40 else "40+")
    show("cos(exec, P17)", lambda r: "nan" if not np.isfinite(r["cos"]) else "<-0.5" if r["cos"] < -0.5 else "<0.5" if r["cos"] < 0.5 else ">=0.5")
    show("own death change", lambda r: "fewer" if r["death"] < -0.25 else "more" if r["death"] > 0.25 else "same")
    show("cargo x enemy Collector", lambda r: ("bat" if r["bat"] > 0 else "empty") + ("/col<3" if r["dcol"] < 3 else "/col>=3"))


if __name__ == "__main__":
    main()
