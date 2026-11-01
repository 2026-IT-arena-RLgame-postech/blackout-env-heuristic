import glob,numpy as np,collections,sys
d='reports/bp/rl4/traj/'
pre=sys.argv[1]
out=[]
for f in sorted(glob.glob(d+pre+'__*.npz')):
    z=np.load(f)
    for j,sd in enumerate(z['sides']):
        if sd!=0: continue
        cls=z['cls'][j]; T=(~np.isnan(z['pos'][j,:,0,0])).sum()
        ev=[]
        for u in range(10):
            c=cls[:T,u]
            for t in np.nonzero((c[1:]==1)&(c[:-1]!=1))[0]+1:
                life=0
                while t+life<T and cls[t+life,u]==1: life+=1
                ev.append((int(t),'A' if u<5 else 'B',u,life))
        ev.sort()
        print(z['seeds'][j],' '.join(f"{t}{tm}{u}:{l}" for t,tm,u,l in ev if t>95)[:700])
        break
