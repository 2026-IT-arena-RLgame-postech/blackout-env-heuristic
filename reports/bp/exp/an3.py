import glob,sys,numpy as np
d=sys.argv[1]
ts=list(range(150,1250,50))
R=[[] for _ in ts];O=[[] for _ in ts]
for f in sorted(glob.glob(d+'/*.npz')):
    z=np.load(f)
    for j,sd in enumerate(z['sides']):
        cls=z['cls'][j]; T=int((cls[:,0]>=0).sum())
        for i,t in enumerate(ts):
            if t<T:
                R[i].append((cls[t,5*sd:5*sd+5]==1).sum()); O[i].append((cls[t,5*(1-sd):5*(1-sd)+5]==1).sum())
print('t   RL   opp  n')
for i,t in enumerate(ts): print(t,'%.2f %.2f %d'%(np.mean(R[i]),np.mean(O[i]),len(R[i])))
