import numpy as np, sys
from collections import Counter
from examples.late_search_why import load_tag, match_facts
from pathlib import Path
from blackout_env.sim.mapdata import HUNTER
out=Path('reports/late_search_why')
opp=sys.argv[1]
P=load_tag(out,'plain',opp); S=load_tag(out,'search',opp)
n=min(len(P['seed']),len(S['seed']))
fP=match_facts(P); fS=match_facts(S)
tied=(fP['race'][:n]==0)&(fS['race'][:n]==0) if opp=='strategic_v23' else np.ones(n,bool)
def sd(p):
    dx=np.maximum(np.maximum(11-p[...,0],p[...,0]-13),0); dy=np.maximum(np.maximum(11-p[...,1],p[...,1]-13),0); return np.hypot(dx,dy)
ts=[50,75,90,100,110,125,150]
for name,D in (('plain',P),('search',S)):
    acc={t:[] for t in ts}; accp={t:[] for t in ts}; nres=[]; held=[]
    for g in np.nonzero(tied)[0]:
        side=int(D['side'][g]); own=list(range(5*side,5*side+5)); op=list(range(5*(1-side),5*(1-side)+5))
        ro=[u for u in own if sd(D['pos'][98,g,u])>1.0]; rp=[u for u in op if sd(D['pos'][98,g,u])>1.0]
        nres.append((len(ro),len(rp)))
        for t in ts:
            acc[t].append(min(sd(D['pos'][t,g,u]) for u in ro) if ro else np.nan)
            accp[t].append(min(sd(D['pos'][t,g,u]) for u in rp) if rp else np.nan)
        if ro:
            u=ro[int(np.argmin([sd(D['pos'][100,g,u]) for u in ro]))]; held.append(D['htyp'][100,g,u])
    print(name, 'reserve dist ours/opp:', ' | '.join(f"{t}:{np.nanmedian(acc[t]):.1f}/{np.nanmedian(accp[t]):.1f}" for t in ts), 'n reserve (ours,opp)', Counter(nres).most_common(4), 'held@100', Counter(held))
    a=np.array(acc[100]); print('    ours@100 quantiles', np.nanquantile(a,[.1,.25,.5,.75,.9]).round(1), ' opp', np.nanquantile(np.array(accp[100]),[.1,.25,.5,.75,.9]).round(1))
