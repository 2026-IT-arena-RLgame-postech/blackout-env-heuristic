"""transform events with >=1 enemy hunter out: where was the enemy hunter, how fast did the transformer come (50 Hz)"""
import glob,numpy as np,collections,sys
d=sys.argv[1]; prefix=sys.argv[2]; n=0
ev={'A':[], 'P':[]}
for f in sorted(glob.glob(d+prefix+'__*.npz')):
    z=np.load(f)
    for j,sd in enumerate(z['sides']):
        T=int((~np.isnan(z['pos'][j,:,0,0])).sum()); n+=1
        cls=z['cls'][j]; pos=z['pos'][j]
        for k,tm in (('A',sd),('P',1-sd)):
            for u in range(5*tm,5*tm+5):
                c=cls[:T,u]
                for t in np.nonzero((c[1:]==1)&(c[:-1]!=1))[0]+1:
                    if t<220: continue
                    eh=np.nonzero(cls[t,5*(1-tm):5*(1-tm)+5]==1)[0]+5*(1-tm)
                    oh=(cls[t,5*tm:5*tm+5]==1).sum()
                    if not len(eh): ev[k].append((oh,0,99,99,99)); continue
                    de=np.hypot(pos[t,eh,0]-12,pos[t,eh,1]-12); i=de.argmin()
                    du=np.hypot(pos[t,eh[i],0]-pos[t,u,0],pos[t,eh[i],1]-pos[t,u,1])
                    # nearest enemy hunter speed proxy: distance moved last 10 ticks
                    ev[k].append((oh,len(eh),de.min(),du,0))
print(prefix,n)
for k in 'AP':
    a=np.array(ev[k])
    if not len(a): continue
    for ne in (0,1,2):
        s=a[a[:,1]==ne]
        if len(s): print(f" {k} transforms with {ne} enemy hunters: n/match {len(s)/n:.2f}"+(f" enemy hunter dist-to-sanct median {np.median(s[:,2]):.1f} q25 {np.quantile(s[:,2],.25):.1f} q75 {np.quantile(s[:,2],.75):.1f}; to transformer med {np.median(s[:,3]):.1f}" if ne else ''))
