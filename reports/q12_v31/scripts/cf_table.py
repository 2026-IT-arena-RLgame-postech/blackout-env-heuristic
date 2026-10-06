"""Q12 (docs §46, search on V30): realised deviation value (cf, vH at +30 s, chosen - base) by deviating unit class x phase x context,
per opponent and pooled.  python reports/q12_v31/scripts/cf_table.py v29,v28,v25"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "examples")
import q11_why as q  # noqa: E402
from blackout_env.sim.mapdata import CARRIER, COLLECTOR, HUNTER  # noqa: E402

OUT = Path("reports/q12_v31")
PH = ((5, 10), (10, 15), (15, 20), (20, 30), (30, 151))
CLS = {COLLECTOR: "Col", HUNTER: "Hun", CARRIER: "Car"}


def rows(opp):
    S = q.load_tag(OUT, "search_b30", opp)
    C = dict(np.load(OUT / f"cf_{opp}_b30.npz"))
    f = q.lsw.root_features(S)
    dvi = q.deviation_dir(S, f)
    g = q.lsw.cf_gains(C)["vH"]
    K = int(C["K"])
    idx = C["root"].astype(int)
    ev = C["ev"].astype(float)  # [R, 2K, 10, keys]: death, kill, ...
    out = []
    for r, i in enumerate(idx):
        t = int(f["tick"][i]) / 25
        ph = next(f"{a}-{b}" for a, b in PH if a <= t < b)
        kind = f["kind"][i]
        u = int(f["unit"][i])
        side = int(S["side"][f["game"][i]])
        if kind == "sam":
            c = "sample"
        else:
            c = CLS[int(f["u_cls"][i])]
            if c == "Col":
                c += "+bat" if f["u_bat"][i] > 0 else ""
        ctx = ""
        if u >= 0:
            gu = u + 5 * side
            dd = ev[r, K:, gu, 0].mean() - ev[r, :K, gu, 0].mean()  # deviating unit's 6 s deaths, chosen - base
            dk = ev[r, K:, gu, 1].mean() - ev[r, :K, gu, 1].mean()
            ctx = ("dies-" if dd < -0.25 else "dies+" if dd > 0.25 else "") + ("kill+" if dk > 0.25 else "kill-" if dk < -0.25 else "")
        out.append(dict(opp=opp, ph=ph, c=c, ctx=ctx or "-", g=float(g[r]), seed=int(S["seed"][f["game"][i]]),
                        own_h=int(f["own_h"][i]), opp_h=int(f["opp_h"][i]), dh=float(f["u_dhunt"][i]),
                        dsanct=float(dvi["d_sanct"][r] if False else dvi["d_sanct"][i]) if u >= 0 else np.nan,
                        dflee=float(dvi["d_flee"][i]) if u >= 0 else np.nan))
    return out, len(S["seed"])


def table(R, key, nm, opps):
    by = defaultdict(lambda: defaultdict(list))
    for r in R:
        by[key(r)][r["opp"]].append(r["g"])
    print(f"-- by {nm}: mean realised vH per deviation (n) per opponent | pooled mean, sum per match (all opps)")
    for k in sorted(by, key=lambda k: -sum(len(v) for v in by[k].values())):
        cells = []
        allv = []
        for o in opps:
            v = by[k].get(o, [])
            allv += v
            cells.append(f"{np.mean(v):+6.2f}±{q.lsw.se(v) if len(v) > 1 else 0:4.2f} ({len(v):3d})" if v else " " * 19)
        print(f"   {str(k):28s} " + "  ".join(cells) + f" | {np.mean(allv):+5.2f}±{q.lsw.se(allv):.2f} n {len(allv)}")


def main():
    opps = sys.argv[1].split(",")
    R = []
    for o in opps:
        rr, nm = rows(o)
        R += rr
        print(f"{o}: {len(rr)} cf roots, {nm} matches")
    print("opponents:", "  ".join(f"{o:19s}" for o in opps))
    table(R, lambda r: r["c"], "class", opps)
    table(R, lambda r: r["ph"], "phase", opps)
    table(R, lambda r: (r["c"], r["ph"]), "class x phase", opps)
    table(R, lambda r: (r["c"], r["ctx"]), "class x own-unit 6 s death/kill change", opps)
    table(R, lambda r: (r["c"], f"own{min(r['own_h'], 3)}v{min(r['opp_h'], 3)}"), "class x Hunters own v opp", opps)
    table(R, lambda r: (r["c"], "near" if r["dh"] <= 2 else "mid" if r["dh"] <= 4 else "far"), "class x enemy-Hunter dist", opps)
    table(R, lambda r: (r["c"], "toward-sanct" if r["dsanct"] > 0.15 else "away-sanct" if r["dsanct"] < -0.15 else "~"),
          "class x radial (vs P17 move)", opps)
    table(R, lambda r: (r["c"], "toward-eH" if r["dflee"] < -0.15 else "away-eH" if r["dflee"] > 0.15 else "~/none"),
          "class x toward enemy Hunter (vs P17 move)", opps)


if __name__ == "__main__":
    main()
