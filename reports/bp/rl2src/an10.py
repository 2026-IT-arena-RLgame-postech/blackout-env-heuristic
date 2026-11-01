import glob,numpy as np,collections
d='reports/bp/rl2/traj/'
def run(prefix,label,t0=150,t1=1500):
    R=collections.defaultdict(float); n=0; tk=0
    for f in sorted(glob.glob(d+prefix+'__*.npz')):
        z=np.load(f)
        for j,(s,sd) in enumerate(zip(z['seeds'],z['sides'])):
            T=(~np.isnan(z['pos'][j,:,0,0])); T=T.sum(); T1=min(T,t1)
            if T1<=t0: continue
            n+=1
            for k,tm in (('A',sd),('P',1-sd)):
                sl=slice(5*tm,5*tm+5); c=z['cls'][j,t0:T1,sl]; p=z['pos'][j,t0:T1,sl]; h=z['held'][j,t0:T1,sl]
                ds=np.hypot(p[...,0]-12,p[...,1]-12)
                colE=(c==0)&(h<0)
                for r in (3,5,8):
                    R[(k,'colnear',r)]+=((colE&(ds<r)).any(1)).sum()
                deficit=((c==1).sum(1)<2)
                R[(k,'def')]+=deficit.sum(); R[(k,'T')]+=len(deficit)
                R[(k,'def_near5')]+=(deficit&(colE&(ds<5)).any(1)).sum()
                R[(k,'def_near8')]+=(deficit&(colE&(ds<8)).any(1)).sum()
                # idle collectors (not moving) count
    print('==',label,n)
    for k in 'AP':
        T=R[(k,'T')]
        print(f" {k}: P(empty collector within 3/5/8 of sanct) {R[(k,'colnear',3)]/T:.2f}/{R[(k,'colnear',5)]/T:.2f}/{R[(k,'colnear',8)]/T:.2f} | deficit frac {R[(k,'def')]/T:.2f}; during deficit: near5 {R[(k,'def_near5')]/R[(k,'def')]:.2f} near8 {R[(k,'def_near8')]/R[(k,'def')]:.2f}")
for p,l in [('bp__B4','RL2(A) v B4(P)'),('B4__B4','B4 v B4'),('bp__C0','RL2(A) v C0(P)'),('B4__C0','B4(A) v C0(P)')]: run(p,l)
