import glob,numpy as np,collections,sys
d=sys.argv[1]; pref=sys.argv[2]
R=collections.defaultdict(float); n=0
for f in sorted(glob.glob(d+pref+'__*.npz')):
    z=np.load(f)
    for j,(s,sd) in enumerate(zip(z['seeds'],z['sides'])):
        T=(~np.isnan(z['pos'][j,:,0,0])).sum(); T1=min(T,2500)
        if T1<=100: continue
        n+=1
        for k,tm in (('A',sd),('P',1-sd)):
            sl=slice(5*tm,5*tm+5); sle=slice(5*(1-tm),5*(1-tm)+5)
            nh=(z['cls'][j,:T1,sl]==1).sum(1); ne=(z['cls'][j,:T1,sle]==1).sum(1)
            for t in range(100,T1-1):
                key=(k,nh[t],ne[t])
                R[key+('t',)]+=1
                if nh[t+1]>nh[t]: R[key+('up',)]+=1
                if nh[t+1]<nh[t]: R[key+('down',)]+=1
print(pref,n)
for hk in (0,1,2):
    for ek in (0,1,2):
        s=''
        for k in 'AP':
            t=R[(k,hk,ek,'t')]
            if t<100: continue
            s+=f" {k}: t/match {t/n:5.0f} up-rate/100ticks {100*R[(k,hk,ek,'up')]/t:5.2f} down {100*R[(k,hk,ek,'down')]/t:5.2f} |"
        print(f"own{hk} enemy{ek}:"+s)
