"""OOD1 (docs §56): how early can a policy see that the opponent commits 4 Hunters (v30_h4 / h4hunt) instead of
V30's 3 + staged reserve?  Plays the first 5 s (decision 125) of ours vs each opponent and prints, per decision
mark, the mean number of enemy units within R tiles of the Hunter sanctuary (and our own), and the enemy Hunter count.
    python id_probe.py --ours p21e --opponent 'V30(ref);v30_h4' --seeds 32"""
import argparse, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "examples"))
from v27_eval import _policy, sanct_dist  # noqa: E402
from diag2 import ALIAS  # noqa: E402
from blackout_env.sim import core  # noqa: E402
from blackout_env.sim.batch import N_AGENT_COLS, N_CH, BatchSim  # noqa: E402
from blackout_env.sim.mapdata import HUNTER, H, W  # noqa: E402
from blackout_env.sim.state import SimState  # noqa: E402
MARKS = (86, 88, 90, 92, 94, 96, 98, 99, 110)
ap = argparse.ArgumentParser(); ap.add_argument("--ours", default="p21e"); ap.add_argument("--opponent", default="V30(ref);v30_h4")
ap.add_argument("--seeds", type=int, default=32); ap.add_argument("--seed0", type=int, default=2600000); ap.add_argument("--r", type=float, default=2.0)
a = ap.parse_args()
for opp_name in a.opponent.split(";"):
    seeds = [s for s in range(a.seed0, a.seed0 + a.seeds) for _ in (0, 1)]; sides = np.array([0, 1] * a.seeds)
    n = len(seeds); ours, opp = _policy(ALIAS.get(a.ours, a.ours)), _policy(ALIAS.get(opp_name, opp_name)); ours.reset(n); opp.reset(n)
    bs = BatchSim([SimState.from_seed(s) for s in seeds])
    bufs = (np.zeros((n, 2, N_CH, H, W), np.float32), np.zeros((n, 2, 4), np.float32), np.zeros((n, 2, 10, N_AGENT_COLS), np.float32))
    mv = np.zeros((n, 10, 2), np.float32); idx = np.arange(n); rec = {}
    for t in range(max(MARKS)):
        bs.render(idx, *bufs)
        for team in (0, 1):
            mine = sides == team
            for pol, sel in ((ours, mine), (opp, ~mine)):
                mv[idx[sel], 5 * team:5 * team + 5] = pol.moves(bs, idx[sel], (team,), tuple(x[sel] for x in bufs))[team]
        bs.step(mv, idx)
        d = t + 1
        if d in MARKS:
            near, hun = np.zeros((n, 2)), np.zeros((n, 2))
            for b in range(n):
                for k, tm in enumerate((sides[b], 1 - sides[b])):
                    u = slice(5 * tm, 5 * tm + 5)
                    near[b, k] = (sanct_dist(bs.uf[b, u]) <= a.r).sum()
                    hun[b, k] = (bs.ui[b, u, core.U_CLS] == HUNTER).sum()
            rec[d] = (near, hun)
    h99 = rec[99][1]
    print(f"{a.ours} vs {opp_name}: enemy Hunters@99 = 4 in {int((h99[:, 1] == 4).sum())}/{n}, ours 4 in {int((h99[:, 0] == 4).sum())}/{n}")
    for d in MARKS:
        near, hun = rec[d]
        print(f"  d{d:3d}: units <= {a.r} tiles of sanctuary ours {near[:, 0].mean():.2f} enemy {near[:, 1].mean():.2f} "
              f"(enemy >= 4: {int((near[:, 1] >= 4).sum())}/{n}) | Hunters ours {hun[:, 0].mean():.2f} enemy {hun[:, 1].mean():.2f}")
