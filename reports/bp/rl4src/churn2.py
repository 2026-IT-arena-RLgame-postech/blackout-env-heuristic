import glob,numpy as np,collections
d='reports/bp/rl4/traj/'
k=0
for f in sorted(glob.glob(d+'B5__*.npz'))[:2]:
    z=np.load(f)
    for j,sd in enumerate(z['sides'][:2]):
        cls=z['cls'][j]; T=(~np.isnan(z['pos'][j,:,0,0])).sum()
        ev=[]
        for u in range(10):
            c=cls[:T,u]
            for t in np.nonzero((c[1:]==1)&(c[:-1]!=1))[0]+1: ev.append((t,u))
        ev.sort(); print(f, 'seed',z['seeds'][j],'T',T)
        print(' ',[(t,u) for t,u in ev if t>90][:40])
        # positions of units at the first post-wipe tie
        for t,u in ev:
            if t>105:
                print('  tie t',t,'unit',u,'pos',z['pos'][j,t,u].round(1),'other team hunters pos',[tuple(z['pos'][j,t,v].round(1)) for v in range(10) if v//5!=u//5 and cls[t,v]==1]); break
