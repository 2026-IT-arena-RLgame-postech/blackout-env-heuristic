import glob,numpy as np,collections
d='reports/bp/rl3/traj/'
S=np.array([12.0,12.0])
def run(prefix,label,t0=110,t1=1500):
    lat=collections.defaultdict(list); near=collections.defaultdict(list); n=0
    deficit=collections.defaultdict(float); ticks=0
    for f in sorted(glob.glob(d+prefix+'__*.npz')):
        z=np.load(f)
        for j,(s,sd) in enumerate(zip(z['seeds'],z['sides'])):
            T=(~np.isnan(z['pos'][j,:,0,0])).sum(); T1=min(T,t1)
            if T1<=t0: continue
            n+=1; ticks+=T1-t0
            for k,tm in (('A',sd),('P',1-sd)):
                sl=slice(5*tm,5*tm+5)
                c=z['cls'][j,:T1,sl]; p=z['pos'][j,:T1,sl]
                h=(c==1).sum(1)
                deficit[k]+=np.maximum(0,2-h[t0:]).sum()
                # for each tick with deficit>0, min distance of non-hunter, non-carrier unit to the sanctuary centre
                dist=np.hypot(p[...,0]-12,p[...,1]-12)
                dist=np.where(c==0,dist,99)
                md=dist.min(1)
                for t in range(t0,T1,10):
                    if h[t]<2: near[k].append(md[t])
                # latency: from drop of h below 2 (h[t]<h[t-1] ) to next increase
                t=t0
                while t<T1-1:
                    if h[t]<h[t-1]:
                        u=t
                        while u<T1 and h[u]<=h[t]: u+=1
                        lat[k].append((u-t) if u<T1 else np.nan)
                        t=u
                    t+=1
    print('==',label,n)
    for k in 'AP':
        l=np.array(lat[k],float); nd=np.array(near[k])
        print(f" {k}: hunter deficit/tick {deficit[k]/ticks:.2f} refill latency med {np.nanmedian(l):.0f} mean {np.nanmean(l):.0f} n/match {len(l)/n:.1f} | min collector dist to sanct while deficit: med {np.median(nd):.1f} frac<4 {np.mean(nd<4):.2f}")
for p,l in [('bp__B5','RL3(A) v B5(P)'),('B5__B5','B5 v B5')]: run(p,l)
