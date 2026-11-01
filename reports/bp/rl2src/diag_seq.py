import sys,collections,math,numpy as np
sys.path.insert(0,'.'); sys.path.insert(0,'examples')
import diag_refill as D
from blackout_env.heuristics import v34_bp
from blackout_env.heuristics.strategic import HUNTER, COLLECTOR
from blackout_env.heuristics.any_angle import pixel_position
from blackout_env.heuristics.v17_planner import MATCH_SECONDS as MS
orig=D.orig
SEQ=collections.defaultdict(list)
def wrap(self, controlled, row_for, states, graphic, walkable):
    g=orig(self, controlled, row_for, states, graphic, walkable)
    ctx=self._teach_ctx; t=int(round((MS-ctx['remaining'])*25))
    shape=graphic.shape[:2]
    hun=sum(self._class_id(states[row_for[n]])==HUNTER for n in controlled)
    a=[n[-1] for n in controlled if any(k=='transform' for _,k in g.get(n,())) and self._class_id(states[row_for[n]])==COLLECTOR]
    SEQ[id(self)].append((t,hun,''.join(a)))
    return g
v34_bp.StrategicHeuristicV34._composition_goals=wrap
D.play(sys.argv[1],sys.argv[2],[8200000,8200001],[0,1],T=700)
for k,v in list(SEQ.items())[:2]:
    print(k); 
    print(' '.join(f"{t}:{h}:{a or '-'}" for t,h,a in v if 200<=t<330))
