import glob,numpy as np,collections
d='reports/bp/rl4/traj/'
def run(prefix,label,t0=110,t1=1500):
    ev=collections.defaultdict(list); n=0; rate=collections.defaultdict(float); tk=0
    for f in sorted(glob.glob(d+prefix+'__*.npz')):
        z=np.load(f)
        for j,(s,sd) in enumerate(zip(z['seeds'],z['sides'])):
            T=(~np.isnan(z['pos'][j,:,0,0])).sum(); T1=min(T,t1)
            if T1<=t0: continue
            n+=1; tk+=T1-t0
            for k,tm in (('A',sd),('P',1-sd)):
                sl=slice(5*tm,5*tm+5); sle=slice(5*(1-tm),5*(1-tm)+5)
                c=z['cls'][j,:T1,sl]; p=z['pos'][j,:T1,sl]; ce=z['cls'][j,:T1,sle]; pe=z['pos'][j,:T1,sle]
                for u in range(5):
                    for t in range(t0,T1):
                        if c[t,u]==1 and c[t-1,u]!=1:
                            hs=(c[t]==1).sum()
                            eh=(ce[t]==1)
                            de=np.hypot(pe[t,:,0]-12,pe[t,:,1]-12)[eh]
                            # time since own hunter count last dropped below 2
                            ev[k].append((hs, eh.sum(), de.min() if len(de) else 99, c[t-1,u], np.hypot(p[t,u,0]-12,p[t,u,1]-12)))
                # time with enemy hunter within 4 of the sanctuary
                eh=(ce==1); de=np.where(eh,np.hypot(pe[...,0]-12,pe[...,1]-12),99).min(1)
                rate[k+'_esent']+=(de[t0:]<4).sum()
    print('==',label,n)
    for k in 'AP':
        a=np.array(ev[k]); 
        print(f" {k}: transforms/match {len(a)/n:.2f}; hunters-after-transform==1 frac {np.mean(a[:,0]==1):.2f}; enemy hunters out at transform {a[:,1].mean():.2f}; enemy hunter within 4 of sanct {np.mean(a[:,2]<4):.2f}; within 8 {np.mean(a[:,2]<8):.2f}; enemy sentry time frac {rate[k+'_esent']/tk:.2f}")
for p,l in [('bp__B5','RL4(A) v B5(P)'),('B5__B5','B5 v B5')]: run(p,l)
