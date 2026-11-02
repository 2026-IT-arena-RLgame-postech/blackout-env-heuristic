import glob,sys,numpy as np
d=sys.argv[1]; ts=list(range(250,751,50))
R=[[] for _ in ts];O=[[] for _ in ts]
for f in sorted(glob.glob(d+'/*.npz')):
    z=np.load(f)
    for j,sd in enumerate(z['sides']):
        T=int((z['cls'][j][:,0]>=0).sum())
        for i,t in enumerate(ts):
            if t>=T: continue
            for team,L in ((sd,R),(1-sd,O)):
                p=z['pos'][j,t,5*team:5*team+5]; c=z['cls'][j,t,5*team:5*team+5]
                dist=np.hypot(p[:,0]-11.5,p[:,1]-11.5)
                L[i].append(((c!=1)&(dist<6)).sum())
print('t  RL_near_sanct(<6)  opp')
for i,t in enumerate(ts): print(t,'%.2f %.2f'%(np.mean(R[i]),np.mean(O[i])))
