"""Why does B4 stay below 2 Hunters vs RL2?  Wrap _composition_goals of B4 and log per decision."""
import sys,collections,numpy as np
sys.path.insert(0,'.'); sys.path.insert(0,'examples')
import torch; torch.set_num_threads(1)
from v27_eval import _policy
from blackout_env.sim import core
from blackout_env.sim.batch import N_AGENT_COLS, N_CH, BatchSim
from blackout_env.sim.mapdata import H, W
from blackout_env.sim.state import SimState
from blackout_env.heuristics import v34_bp
from blackout_env.heuristics.strategic import HUNTER, COLLECTOR
LOG=collections.Counter()
orig=v34_bp.StrategicHeuristicV34._composition_goals
def wrap(self, controlled, row_for, states, graphic, walkable):
    g=orig(self, controlled, row_for, states, graphic, walkable)
    ctx=self._teach_ctx
    from blackout_env.heuristics.v17_planner import MATCH_SECONDS as MS
    t=int(round((MS-ctx['remaining'])*25))
    if 110<=t<800:
        cls={n:self._class_id(states[row_for[n]]) for n in controlled}
        hun=sum(c==HUNTER for c in cls.values())
        if hun<2:
            nt=sum(1 for n,v in g.items() if any(k=='transform' for _,k in v))
            cand=[n for n in controlled if cls[n]==COLLECTOR]
            empty=[n for n in cand if not self._is_holding(states[row_for[n]])]
            eh=sum(e.cls==HUNTER and not e.in_base for e in ctx['enemies'])
            LOG['dec_deficit']+=1
            LOG[f'h{hun}_transform{nt}_cand{min(len(empty),3)}_enemyH{eh}']+=1
            LOG['hold_n']+=hun
    LOG['dec_all']+=1
    return g
v34_bp.StrategicHeuristicV34._composition_goals=wrap
def play(ours,opp,seeds,sides,T=800):
    F32=np.float32; n=len(seeds); sides=np.asarray(sides)
    po,pp=_policy(ours),_policy(opp); po.reset(n); pp.reset(n)
    bs=BatchSim([SimState.from_seed(int(s)) for s in seeds])
    bufs=(np.zeros((n,2,N_CH,H,W),F32),np.zeros((n,2,4),F32),np.zeros((n,2,10,N_AGENT_COLS),F32))
    moves=np.zeros((n,10,2),F32)
    for t in range(T):
        live=np.nonzero(~bs.over)[0]
        if not len(live): break
        bs.render(live,*bufs); sub=tuple(x[:len(live)] for x in bufs)
        for team in (0,1):
            mine=sides[live]==team
            for pol,sel in ((po,mine),(pp,~mine)):
                if not sel.any(): continue
                s_in=sub if sel.all() else tuple(x[sel] for x in sub)
                moves[live[sel],5*team:5*team+5]=pol.moves(bs,live[sel],(team,),s_in)[team]
        bs.step(moves,live)
if __name__=='__main__':
    from blackout_env.heuristics.params import resolve_descriptor as rd
    ours,opp=sys.argv[1],sys.argv[2]
    seeds=list(range(8200000,8200012))
    play(ours,opp,seeds,[i%2 for i in range(12)])
    for k,v in sorted(LOG.items()): print(k,v)
