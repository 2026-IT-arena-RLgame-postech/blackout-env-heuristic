"""Q10: Hunters at decision 99 (ours / theirs) for V29 vs each opponent, sim, 32 seeds x 2 sides (seed 1800000+).
Also counts how often V29's staged reserve joins.  Nets on CPU (BLACKOUT_NETS as v27_eval.py)."""
import sys
from collections import Counter
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "examples"))
from v27_eval import _policy
from blackout_env.sim.batch import BatchSim, N_AGENT_COLS, N_CH
from blackout_env.sim.mapdata import HUNTER, H, W
from blackout_env.sim.state import SimState

V29 = "strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve"
V28 = "strategic_v28:reserve_stage=0.5,trade_hold=advance"
OPPS = [V28, "p17", "p16", "strategic_v27:reserve_stage=1.0", "strategic_v27:reserve_stage=1.5", "strategic_v25",
        "p14b", "strategic_v23", "strategic_v28:reserve_stage=0.75,trade_hold=advance", V29]
for ours_name in (V29, V28):
    for opp_name in OPPS:
        seeds = [1800000 + i for i in range(32) for _ in (0, 1)]
        sides = np.array([i % 2 for i in range(64)])
        n = len(seeds)
        ours, opp = _policy(ours_name), _policy(opp_name)
        ours.reset(n); opp.reset(n)
        bs = BatchSim([SimState.from_seed(s) for s in seeds])
        bufs = (np.zeros((n, 2, N_CH, H, W), np.float32), np.zeros((n, 2, 4), np.float32), np.zeros((n, 2, 10, N_AGENT_COLS), np.float32))
        mv = np.zeros((n, 10, 2), np.float32)
        live = np.arange(n)
        for t in range(99):
            bs.render(live, *bufs)
            for team in (0, 1):
                for pol, sel in ((ours, sides == team), (opp, sides != team)):
                    if sel.any():
                        mv[live[sel], 5 * team:5 * team + 5] = pol.moves(bs, live[sel], (team,), tuple(x[sel] for x in bufs))[team]
            bs.step(mv, live)
        c = Counter()
        for b in range(n):
            o = 5 * sides[b]; p = 5 - o
            c[(int((bs.ui[b, o:o + 5, 0] == HUNTER).sum()), int((bs.ui[b, p:p + 5, 0] == HUNTER).sum()))] += 1
        print(f"{ours_name[:13]:13s} vs {opp_name[:40]:40s} Hunters@99 (ours, theirs): {dict(sorted(c.items()))}", flush=True)
