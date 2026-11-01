"""Mixed-rate sim: team A queried every tick (50 Hz) or every 2nd tick (25 Hz, move held), same for team B.
Usage: hz_test.py <A desc> <A hz> <B desc> <B hz> <n games> <seed0>"""
import sys, numpy as np
sys.path.insert(0,'.'); sys.path.insert(0,'examples')
from blackout_env.sim.batch import N_AGENT_COLS, N_CH, BatchSim
from blackout_env.sim.mapdata import H, W
from blackout_env.sim.state import SimState
from blackout_env.sim import core
from v27_eval import _policy
A, ha, B, hb, n, s0 = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
res=[]
for swap in (False, True):
    pa, pb = _policy(A), _policy(B); pa.reset(n); pb.reset(n)
    bs = BatchSim([SimState.from_seed(s0 + i) for i in range(n)]); bs.ticks_per_decision = 1
    F=np.float32; g=np.zeros((n,2,N_CH,H,W),F); t=np.zeros((n,2,4),F); ag=np.zeros((n,2,10,N_AGENT_COLS),F); mv=np.zeros((n,10,2),F)
    ta, tb = (1, 0) if swap else (0, 1)
    k = 0
    first_h = np.full((n, 2), -1)
    while True:
        live = np.nonzero(~bs.over)[0]
        if not len(live) or k > 21000: break
        if k % 2 == 0 or ha == 50 or hb == 50:
            bs.render(live, g, t, ag); sub = tuple(x[:len(live)] for x in (g, t, ag))
        if ha == 50 or k % 2 == 0:
            mv[live, 5*ta:5*ta+5] = pa.moves(bs, live, (ta,), sub)[ta]
        if hb == 50 or k % 2 == 0:
            mv[live, 5*tb:5*tb+5] = pb.moves(bs, live, (tb,), sub)[tb]
        bs.step(mv, live); k += 1
        cls = bs.ui[:, :, core.U_CLS]
        for tm in (0, 1):
            has = (cls[:, 5*tm:5*tm+5] == 1).any(1) & (first_h[:, tm] < 0)
            first_h[has, tm] = k
    sc = bs.scores
    for i in range(n):
        res.append((sc[i, ta] - sc[i, tb], first_h[i, ta], first_h[i, tb]))
m = np.array([r[0] for r in res], float)
print(f"{A[:20]}@{ha} vs {B[:20]}@{hb}: margin {m.mean():+.1f}±{m.std(ddof=1)/np.sqrt(len(m)):.1f}  win {np.mean(m>0)+0.5*np.mean(m==0):.2f}  n={len(m)}  first-hunter tick A {np.mean([r[1] for r in res]):.0f} B {np.mean([r[2] for r in res]):.0f}")
