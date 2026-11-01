import glob,numpy as np,collections,sys
d=sys.argv[1]; pref=sys.argv[2]
R=collections.defaultdict(float); n=0
def zone(p,tm):
    eb=(20,1) if tm==0 else (1,20); ob=(1,20) if tm==0 else (20,1)
    ds=np.hypot(p[...,0]-12,p[...,1]-12); de=np.hypot(p[...,0]-eb[0],p[...,1]-eb[1]); do=np.hypot(p[...,0]-ob[0],p[...,1]-ob[1])
    z=np.full(ds.shape,3)  # other
    z=np.where(do<7,2,z); z=np.where(de<7,1,z); z=np.where(ds<3,0,z)
    return z
for f in sorted(glob.glob(d+pref+'__*.npz')):
    z=np.load(f)
    for j,(s,sd) in enumerate(zip(z['seeds'],z['sides'])):
        T=(~np.isnan(z['pos'][j,:,0,0])).sum(); T1=min(T,1500)
        if T1<=150: continue
        n+=1
        for k,tm in (('A',sd),('P',1-sd)):
            sl=slice(5*tm,5*tm+5); sle=slice(5*(1-tm),5*(1-tm)+5)
            c=z['cls'][j,150:T1,sl]; p=z['pos'][j,150:T1,sl]; ce=z['cls'][j,150:T1,sle]
            h=(c==1); nh=h.sum(1); ne=(ce==1).sum(1)
            zn=zone(p,tm)
            mv=np.zeros_like(h); sp=np.hypot(*(np.diff(p,axis=0).transpose(2,0,1))); mv[1:]=sp>0.02
            for hk in (1,2):
                for ek in (0,1,2):
                    sel=(nh==hk)&(ne==ek)
                    if not sel.any(): continue
                    hh=h&sel[:,None]
                    R[(k,hk,ek,'t')]+=hh.sum()
                    for zz in range(4): R[(k,hk,ek,zz)]+=(hh&(zn==zz)).sum()
                    R[(k,hk,ek,'mv')]+=(hh&mv).sum()
print(pref,n)
for hk in (1,2):
    for ek in (0,1,2):
        for k in 'AP':
            t=R[(k,hk,ek,'t')]
            if t<200: continue
            print(f" own{hk} enemy{ek} {k}: hunter-ticks/match {t/n:6.0f} zones sanct/ebase/own/other {R[(k,hk,ek,0)]/t:.2f}/{R[(k,hk,ek,1)]/t:.2f}/{R[(k,hk,ek,2)]/t:.2f}/{R[(k,hk,ek,3)]/t:.2f} moving {R[(k,hk,ek,'mv')]/t:.2f}")
