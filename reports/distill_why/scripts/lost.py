import sys, numpy as np
sys.path.insert(0,'examples'); sys.path.insert(0,'.')
from d1_why import load, outcome_rows, HUNTER, COLLECTOR, CARRIER
out=sys.argv[1]; opp=sys.argv[2]
for o in sys.argv[3].split(';'):
    D=load(out,o,opp); F=outcome_rows(D); side=D['side'].astype(int); n=len(side)
    us=np.stack([np.arange(5)+5*s for s in side]); op=np.stack([np.arange(5)+5*(1-s) for s in side])
    lost=F['margin']<-20
    def cnt(t,grp,c): return np.array([(D['cls'][min(t,D['cls'].shape[0]-1),b,grp[b]]==c).sum() for b in range(n)])
    h4=F['h99']==4
    s=f"{o:7s} lost {lost.sum():2d}/64 | lost|h99=4 {lost[h4].mean():.2f} (n={h4.sum()}) lost|h99=3 {lost[~h4].mean():.2f} (n={(~h4).sum()})"
    for t in (250,375,500,750):
        oh=cnt(t,op,HUNTER); mh=cnt(t,us,HUNTER); mc=cnt(t,us,COLLECTOR)
        s+=f" | t{t//25}s H us/opp lost {mh[lost].mean():.1f}/{oh[lost].mean():.1f} won {mh[~lost].mean():.1f}/{oh[~lost].mean():.1f}"
    print(s)
