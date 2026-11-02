import glob,sys,numpy as np
for d in sys.argv[1:]:
    ts=[400,800,1200,1600]; M={t:[[],[]] for t in ts}
    for f in sorted(glob.glob(d+'/*.npz')):
        z=np.load(f)
        for j,sd in enumerate(z['sides']):
            T=int((z['cls'][j][:,0]>=0).sum()); sc=z['score'][j]
            for t in ts:
                tt=min(t,T-1); M[t][0].append(sc[tt,sd]); M[t][1].append(sc[tt,1-sd])
    print(d.split('/')[-1],' '.join(f"{t}: RL {np.mean(M[t][0]):.0f} opp {np.mean(M[t][1]):.0f} |" for t in ts))
