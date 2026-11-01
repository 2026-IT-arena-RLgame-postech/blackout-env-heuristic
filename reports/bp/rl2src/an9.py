import glob,numpy as np,collections
d='reports/bp/rl2/traj/'
def run(prefix,label,t0=115):
    R=collections.defaultdict(list); n=0
    for f in sorted(glob.glob(d+prefix+'__*.npz')):
        z=np.load(f)
        for j,(s,sd) in enumerate(zip(z['seeds'],z['sides'])):
            T=(~np.isnan(z['pos'][j,:,0,0])).sum(); n+=1
            for k,tm in (('A',sd),('P',1-sd)):
                sl=slice(5*tm,5*tm+5); c=z['cls'][j,:T,sl]
                for u in range(5):
                    t=t0
                    while t<T:
                        if c[t,u]==1 and c[t-1,u]!=1:
                            e=t
                            while e<T and c[e,u]==1: e+=1
                            R[k].append(e-t if e<T else 99999)
                            t=e
                        t+=1
    print('==',label,n)
    for k in 'AP':
        l=np.array(R[k]); 
        print(f" {k}: post-opening transforms/match {len(l)/n:.2f}  life<=5: {np.sum(l<=5)/n:.2f}  6-50: {np.sum((l>5)&(l<=50))/n:.2f}  51-300: {np.sum((l>50)&(l<=300))/n:.2f} >300 or alive: {np.sum(l>300)/n:.2f}")
for p,l in [('bp__B4','RL2(A) v B4(P)'),('B4__B4','B4 v B4'),('bp__C0','RL2(A) v C0(P)'),('B4__C0','B4(A) v C0(P)')]: run(p,l)
