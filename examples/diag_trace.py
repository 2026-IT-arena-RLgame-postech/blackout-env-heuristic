"""Trace B5's transform walkers in 1v1 deficits: goal present?, walker distance to sanctuary, cap-hold, enemy Hunter distance."""
import sys,math,numpy as np
sys.path.insert(0,'.'); sys.path.insert(0,'examples')
import diag_refill as D
from blackout_env.heuristics import v34_bp
from blackout_env.heuristics.strategic import HUNTER, COLLECTOR
from blackout_env.heuristics.any_angle import pixel_position
from blackout_env.heuristics.v17_planner import MATCH_SECONDS as MS
orig=D.orig
T0,T1,STEP=int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])
def wrap(self, controlled, row_for, states, graphic, walkable):
    g=orig(self, controlled, row_for, states, graphic, walkable)
    ctx=self._teach_ctx; t=int(round((MS-ctx['remaining'])*25)); shape=graphic.shape[:2]
    if T0<=t<T1 and t%STEP==0:
        hun=[n for n in controlled if self._class_id(states[row_for[n]])==HUNTER]
        eh=[pixel_position(states[e.row],shape) for e in ctx['enemies'] if e.cls==HUNTER and not e.in_base]
        w=[]
        for n in controlled:
            s=states[row_for[n]]
            if self._class_id(s)==COLLECTOR:
                p=pixel_position(s,shape); tr=any(k=='transform' for _,k in g.get(n,()))
                w.append(f"{n[-1]}{'T' if tr else '-'}{'H' if n in getattr(self,'_cap_hold',()) else ''}{'C' if n in self.committed else ''}:{math.hypot(p[0]-12,p[1]-12):.0f}")
        print(t,'H',len(hun),'enemyH',[f"({a:.0f},{b:.0f})" for a,b in eh],' '.join(w),flush=True)
    return g
v34_bp.StrategicHeuristicV34._composition_goals=wrap
D.play(sys.argv[1],sys.argv[2],[8200000],[0],T=T1)
