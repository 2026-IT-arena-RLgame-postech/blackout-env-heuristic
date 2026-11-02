import glob,sys,numpy as np
for d in sys.argv[1:]:
    ts=[200,300,400,600,800,1000,1400,2000]; M={t:[] for t in ts}; ends=[]
    for f in sorted(glob.glob(d+'/*.npz')):
        z=np.load(f)
        for j,sd in enumerate(z['sides']):
            T=int((z['cls'][j][:,0]>=0).sum()); sc=z['score'][j]
            for t in ts:
                tt=min(t,T-1); M[t].append(int(sc[tt,sd])-int(sc[tt,1-sd]))
            ends.append(T)
    print(d.split('/')[-1],' '.join(f"{t}:{np.mean(M[t]):+.1f}" for t in ts),'len',np.mean(ends))
