import numpy as np, sys, json
from collections import Counter
from examples.late_search_why import load_tag, match_facts
from pathlib import Path
from blackout_env.sim import core
from blackout_env.sim.mapdata import HUNTER
out=Path('reports/late_search_why')
opp=sys.argv[1]
P=load_tag(out,'plain',opp); S=load_tag(out,'search',opp)
n=min(len(P['seed']),len(S['seed']))
fP=match_facts(P); fS=match_facts(S)
tied=(fP['race'][:n]==0)&(fS['race'][:n]==0) if opp=='strategic_v23' else np.ones(n,bool)
def second(D,g,side):
    """first post-race Hunter transform tick per team (after the opening cancellation): returns (ours, opp, hunters held at 250)"""
    ev=D['events']; ev=ev[(ev[:,0]==g)]
    fo=fp=10**6
    first=min(int(np.nonzero((D['cls'][:,g]==HUNTER).any(-1))[0][0]) if (D['cls'][:,g]==HUNTER).any() else 10**6, 10**6)
    for e in ev:
        _,t,k,a,b,c,d=e
        if k==core.EV_TRANSFORM and b==HUNTER and t>first+8:
            if (a<5)==(side==0): fo=min(fo,t)
            else: fp=min(fp,t)
    return fo,fp
res={}
for name,D in (('plain',P),('search',S)):
    r=[]
    for g in np.nonzero(tied)[0]:
        fo,fp=second(D,g,int(D['side'][g])); r.append((fo,fp))
    r=np.array(r); res[name]=r
    w=(r[:,0]<r[:,1]).sum(); l=(r[:,0]>r[:,1]).sum(); t=(r[:,0]==r[:,1]).sum()
    print(name, f"second race w/t/l {w}/{t}/{l}  median ours {np.median(r[:,0][r[:,0]<10**6]):.0f} opp {np.median(r[:,1][r[:,1]<10**6]):.0f}")
ls=fS['lead'][:n][tied]-fP['lead'][:n][tied]
sw=np.sign(res['search'][:,1]-res['search'][:,0]); pw=np.sign(res['plain'][:,1]-res['plain'][:,0])
for a in (-1,0,1):
  for b in (-1,0,1):
    m=(pw==a)&(sw==b)
    if m.any(): print(f" second race plain {a:+d} -> search {b:+d}: {m.sum():3d} diff {ls[m].mean():+6.1f} sum share {ls[m].sum()/ls.sum():+.2f}")
print("--- second-race winner details (our first post-opening transform)")
def sd(p):
    dx=np.maximum(np.maximum(11-p[...,0],p[...,0]-13),0); dy=np.maximum(np.maximum(11-p[...,1],p[...,1]-13),0); return np.hypot(dx,dy)
for name,D in (('plain',P),('search',S)):
    rows=[]
    for g in np.nonzero(tied)[0]:
        side=int(D['side'][g]); ev=D['events']; ev=ev[ev[:,0]==g]
        first=int(np.nonzero((D['cls'][:,g]==HUNTER).any(-1))[0][0])
        tr=[(int(e[1]),int(e[3])) for e in ev if e[2]==core.EV_TRANSFORM and e[4]==HUNTER and e[1]>first+8 and (e[3]<5)==(side==0)]
        if not tr: continue
        t,u=tr[0]
        died=[int(e[1]) for e in ev if e[2]==core.EV_KILL and e[4]==u and e[1]<t]
        origin='respawned' if died else 'fresh'
        # was it carrying at t-25? held battery at transform-1
        carried=int(D['htyp'][t-1,g,u])
        d105=float(sd(D['pos'][first+8,g,u]))
        # decision when it started heading (dist decreasing monotonic) - approximate: last tick before t where it was farther than at t-1 by >... skip
        rows.append((t,origin,carried,d105))
    t=np.array([r[0] for r in rows]); o=Counter(r[1] for r in rows); c=Counter(r[2] for r in rows)
    print(name, 'n',len(rows),'median t',np.median(t), dict(o), 'held before transform', dict(c), 'dist@first+8 median', np.median([r[3] for r in rows]).round(2))
    for orig in ('respawned','fresh'):
        tt=[r[0] for r in rows if r[1]==orig]; print('   ',orig, len(tt), 'median t', np.median(tt) if tt else None, 'dist', np.median([r[3] for r in rows if r[1]==orig]).round(2) if tt else None)
