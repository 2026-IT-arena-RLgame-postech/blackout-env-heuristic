"""trace3.py <descriptor> <net> <seed> <side> <tmax>: event timeline + unit snapshots"""
import sys, numpy as np
sys.path.insert(0, "examples")
import v27_eval as V
from blackout_env.sim import core, batch as _batch
from blackout_env.sim.batch import N_AGENT_COLS, N_CH, BatchSim
from blackout_env.sim.mapdata import H, W
from blackout_env.sim.state import SimState
desc, netp, seed, sd, tmax = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
every = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
n = 1
ours, opp = V._policy(desc), V._policy("net:" + netp)
ours.reset(n); opp.reset(n)
bs = BatchSim([SimState.from_seed(seed)])
F32 = np.float32
bufs = (np.zeros((n, 2, N_CH, H, W), F32), np.zeros((n, 2, 4), F32), np.zeros((n, 2, 10, N_AGENT_COLS), F32))
moves = np.zeros((n, 10, 2), F32)
NM = {0:"EV_PICK",1:"DEP",2:"KILL",3:"TRANSF",4:"ABS",5:"SPAWN",6:"LOST",7:"END",8:"STEAL"}
def lab(u): return ("D" if (u < 5) == (sd == 0) else "R") + str(u % 5)
for t in range(int(tmax * 50 // _batch.DEFAULT_TPD)):
    live = np.nonzero(~bs.over)[0]
    if not len(live): break
    bs.render(live, *bufs)
    for team in (0, 1):
        mine = np.full(1, sd == team)
        for pol, sel in ((ours, mine), (opp, ~mine)):
            if not sel.any(): continue
            moves[live[sel], 5*team:5*team+5] = pol.moves(bs, live[sel], (team,), tuple(x[:1] for x in bufs))[team]
    pre = bs.uf.copy(); 
    bs.step(moves, live)
    ne = int(bs.g[0, core.G_NEV]); sec = (t + 1) * _batch.DEFAULT_TPD / 50
    for e in bs.ev[0, :ne]:
        k = int(e[1])
        if k in (2, 3, 1, 8):
            extra = ""
            if k == 2: extra = f"killer={lab(e[2]) if e[2]>=0 else 'env'} victim={lab(e[3])} at {pre[0,e[3]][:2].round(1)}"
            elif k == 3: extra = f"{lab(e[2])} ->cls{e[3]}"
            else: extra = f"{lab(e[2])} a={e[3]},{e[4]},{e[5]}"
            print(f"{sec:6.2f} {NM[k]} {extra}")
    if abs((sec / every) - round(sec / every)) < 1e-6:
        cls = bs.ui[0, :, core.U_CLS]; pos = bs.uf[0, :, :2]
        s = " ".join(f"{lab(u)}{'CHa'[cls[u]]}({pos[u,0]:.0f},{pos[u,1]:.0f})" for u in range(10))
        print(f"{sec:6.2f} SNAP sc={bs.scores[0, sd]}/{bs.scores[0,1-sd]} {s}")
