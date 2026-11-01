import glob,numpy as np,sys
d=sys.argv[1]; prefix=sys.argv[2]; n=0
R={'A':[], 'P':[]}
for f in sorted(glob.glob(d+prefix+'__*.npz')):
    z=np.load(f)
    for j,sd in enumerate(z['sides']):
        T=int((~np.isnan(z['pos'][j,:,0,0])).sum()); n+=1
        cls=z['cls'][j]; pos=z['pos'][j]
        for k,tm in (('A',sd),('P',1-sd)):
            for u in range(5*tm,5*tm+5):
                c=cls[:T,u]
                for t in np.nonzero((c[1:]==1)&(c[:-1]!=1))[0]+1:
                    if t<300: continue
                    eh=np.nonzero(cls[t,5*(1-tm):5*(1-tm)+5]==1)[0]+5*(1-tm)
                    if not len(eh): continue
                    row=[np.hypot(pos[t-l,u,0]-12,pos[t-l,u,1]-12) for l in (0,25,50,100,200)]
                    # enemy nearest hunter dist to sanct at t-50
                    ed=min(np.hypot(pos[t-50,e,0]-12,pos[t-50,e,1]-12) if cls[t-50,e]==1 else 99 for e in eh)
                    life=0
                    while t+life<T and cls[t+life,u]==1: life+=1
                    R[k].append(row+[ed,life/50])
print(prefix,n)
for k in 'AP':
    a=np.array(R[k]); print(f" {k} n/match {len(a)/n:.2f} transformer dist to sanct at t-0,-.5,-1,-2,-4s median:",np.round(np.median(a[:,:5],0),1),f"enemy hunter dist at -1s med {np.median(a[:,5]):.1f}; transformer life median {np.median(a[:,6]):.1f}s frac life<1s {np.mean(a[:,6]<1):.2f}")
