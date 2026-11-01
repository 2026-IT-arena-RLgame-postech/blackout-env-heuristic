import glob,numpy as np,collections
d='reports/bp/rl2/traj/'
def run(prefix,label,t0=110):
    life=collections.defaultdict(list); loc=collections.defaultdict(list); n=0
    for f in sorted(glob.glob(d+prefix+'__*.npz')):
        z=np.load(f)
        for j,(s,sd) in enumerate(zip(z['seeds'],z['sides'])):
            T=(~np.isnan(z['pos'][j,:,0,0])).sum(); n+=1
            for k,tm in (('A',sd),('P',1-sd)):
                sl=slice(5*tm,5*tm+5); c=z['cls'][j,:T,sl]; p=z['pos'][j,:T,sl]
                ebase=(20,1) if tm==0 else (1,20)
                for u in range(5):
                    start=None
                    for t in range(1,T):
                        if c[t,u]==1 and c[t-1,u]!=1: start=t
                        if c[t,u]!=1 and c[t-1,u]==1 and start is not None and start>=t0-20:
                            life[k].append(t-start)
                            q=p[t-1,u]
                            loc[k].append((np.hypot(q[0]-12,q[1]-12), np.hypot(q[0]-ebase[0],q[1]-ebase[1])))
                            start=None
    print('==',label,n)
    for k in 'AP':
        l=np.array(life[k]); a=np.array(loc[k])
        print(f" {k}: hunter deaths/match(after t0) {len(l)/n:.2f} lifetime med {np.median(l):.0f} mean {l.mean():.0f}; death near sanct(<4) {np.mean(a[:,0]<4):.2f} near enemy base (<7) {np.mean(a[:,1]<7):.2f}")
for p,l in [('bp__B4','RL2(A) v B4(P)'),('B4__B4','B4 v B4'),('bp__C0','RL2(A) v C0(P)'),('B4__C0','B4(A) v C0(P)')]: run(p,l)
