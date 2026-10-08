"""Q10: after the opening trades (decision 101-130), where is our remaining Hunter and does the enemy's reserve get
in?  V29 'always' (join) vs V28 against V27(1.5) / V25, from q7_why recordings in reports/q10_v29/diag_join."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples"))
from q7_why import load, sanct_dist, HUNTER, COLLECTOR

OUT = Path(__file__).resolve().parents[1] / "diag_join"
for opp in ("strategic_v27:reserve_stage=1.5", "strategic_v25"):
    for ours in ("strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=always",
                 "strategic_v28:reserve_stage=0.5,trade_hold=advance"):
        D = load(OUT, ours, opp)
        n = len(D["seed"])
        hd, ed, opp_first, our_h = [], [], [], []
        for b in range(n):
            s = int(D["side"][b]); us, op = np.arange(5) + 5 * s, np.arange(5) + 5 * (1 - s)
            c = D["cls"][:, b]; p = D["pos"][:, b].astype(np.float32)
            h = [u for u in us if c[104, u] == HUNTER]
            our_h.append(len(h))
            hd.append(np.mean([sanct_dist(p[110, u]) for u in h]) if h else np.nan)
            ec = [u for u in op if c[101, u] == COLLECTOR]
            ed.append(min(sanct_dist(p[101:131, u]).min() for u in ec) if ec else np.nan)
            t = [t for t in range(102, min(400, c.shape[0])) if (c[t, op] == HUNTER).any()]
            opp_first.append(t[0] if t else 9999)
        opp_first = np.array(opp_first)
        print(f"{ours[-22:]:24s} vs {opp:32s} our Hunters@104 {np.bincount(our_h, minlength=4).tolist()} "
              f"our Hunter->sanctuary @110 median {np.nanmedian(hd):.2f}  enemy Collector closest 101-130 median {np.nanmedian(ed):.2f}  "
              f"enemy Hunter by 130: {(opp_first <= 130).mean():.2f} by 200: {(opp_first <= 200).mean():.2f}")
