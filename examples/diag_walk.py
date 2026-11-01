"""Fate of B4's sanctuary walkers (transform goals) after the opening, vs RL2 / B4."""
import sys,collections,math,numpy as np
sys.path.insert(0,'.'); sys.path.insert(0,'examples')
import diag_refill as D  # reuse play; but replace wrapper
from blackout_env.heuristics import v34_bp
from blackout_env.heuristics.strategic import HUNTER, COLLECTOR
from blackout_env.heuristics.any_angle import pixel_position
from blackout_env.heuristics.v17_planner import MATCH_SECONDS as MS
v34_bp.StrategicHeuristicV34._composition_goals=D.orig
orig=D.orig
EP=[]; ST={}
def wrap(self, controlled, row_for, states, graphic, walkable):
    g=orig(self, controlled, row_for, states, graphic, walkable)
    ctx=self._teach_ctx; t=int(round((MS-ctx['remaining'])*25))
    st=ST.setdefault(id(self),{})  # name -> dict(start,t,pos)
    shape=graphic.shape[:2]
    eh=[pixel_position(states[e.row],shape) for e in ctx['enemies'] if e.cls==HUNTER and not e.in_base]
    for n in controlled:
        s=states[row_for[n]]; cls=self._class_id(s); pos=pixel_position(s,shape)
        a=any(k=='transform' for _,k in g.get(n,())) and cls==COLLECTOR
        e=st.get(n)
        if e is not None:
            jump=math.hypot(pos[0]-e['pos'][0],pos[1]-e['pos'][1])>2.0
            if cls==HUNTER: EP.append((e['start'],t,'success',e['d0'],e['minH']));st.pop(n)
            elif jump: EP.append((e['start'],t,'died',e['d0'],e['minH']));st.pop(n)
            elif not a: EP.append((e['start'],t,'abort',e['d0'],e['minH']));st.pop(n)
        if a and n not in st and t>=110:
            d0=math.hypot(pos[0]-12,pos[1]-12)
            st[n]=dict(start=t,pos=pos,d0=d0,minH=99)
        if n in st:
            st[n]['pos']=pos
            if eh: st[n]['minH']=min(st[n]['minH'],min(math.hypot(pos[0]-h[0],pos[1]-h[1]) for h in eh))
    return g
v34_bp.StrategicHeuristicV34._composition_goals=wrap
if __name__=='__main__':
    ours,opp=sys.argv[1],sys.argv[2]
    D.play(ours,opp,list(range(8200000,8200012)),[i%2 for i in range(12)],T=1000)
    EP2=[e for e in EP if e[0]<800]
    c=collections.Counter(e[2] for e in EP2)
    print(c, 'n',len(EP2))
    for k in ('success','died','abort'):
        x=[e for e in EP2 if e[2]==k]
        if x: print(k,'dur med',np.median([e[1]-e[0] for e in x]),'d0 med',np.median([e[3] for e in x]),'min dist to enemy H med',np.median([e[4] for e in x]))
