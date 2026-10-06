import sys, glob, numpy as np
sys.path.insert(0,'examples'); sys.path.insert(0,'.')
from d1_why import load, sanct_dist, COLLECTOR, F32
from pathlib import Path
out=sys.argv[1]
opps=sorted({Path(p).name.split('__')[1] for p in glob.glob(out+'/raw/*.npz')})
C=np.array([12.0,12.0])
print(f"{'opponent':36s} rows/g(4th Coll <=1.5 tiles, t88-98)  toward  stand  away(cos<0)  side")
agg=np.zeros(4); W=0
for o in opps:
    D=load(out,'v30',o); side=D['side'].astype(int); n=len(side); mv=D['ownmv'].astype(F32)
    cnt=np.zeros(4)
    for b in range(n):
        us=np.arange(5)+5*side[b]
        for t in range(88,99):
            c=D['cls'][t,b,us]; p=D['pos'][t,b,us].astype(F32); d=sanct_dist(p)
            idx=np.nonzero(c==COLLECTOR)[0]
            if len(idx)<4: continue
            u=idx[np.argsort(d[idx])[3]]
            if d[u]>1.5: continue
            m=mv[t,b,u]; nm=np.linalg.norm(m)
            if nm<1e-3: cnt[1]+=1; continue
            to=(C-p[u]); to/=max(np.linalg.norm(to),1e-6); cs=float(m@to)/nm
            cnt[0 if cs>0.5 else 2 if cs<0 else 3]+=1
    w=0.35 if o=='mixture' else 0.5/19
    agg+=w*cnt/n; W+=w
    print(f"{o:36s} {cnt.sum()/n:8.2f}  " + "  ".join(f"{x/n:6.2f}" for x in cnt[[0,1,2,3]]))
print('pool-weighted per match (toward, stand, away, side):', np.round(agg,2), 'of simulated weight', round(W,2))
