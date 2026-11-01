import glob,numpy as np,collections,sys
d=sys.argv[1]; pref=sys.argv[2]
R=collections.defaultdict(float); n=0
for f in sorted(glob.glob(d+pref+'__*.npz')):
    z=np.load(f)
    for j,(s,sd) in enumerate(zip(z['seeds'],z['sides'])):
        T=(~np.isnan(z['pos'][j,:,0,0])).sum(); n+=1
        kk=z['kills'][z['kills'][:,0]==j]
        for k,tm in (('A',sd),('P',1-sd)):
            sl=slice(5*tm,5*tm+5); c=z['cls'][j,:T,sl]
            ends=((c[1:]!=1)&(c[:-1]==1))
            starts=((c[1:]==1)&(c[:-1]!=1))
            for a,b in ((110,400),(400,800),(800,1500)):
                R[(k,a,'ends')]+=ends[a:b].sum(); R[(k,a,'starts')]+=starts[a:b].sum()
                kv=sum(1 for _,t,kl,v,vc,kc in kk if a<=t<b and vc==1 and ((0 if v<5 else 1)==tm))
                R[(k,a,'killed')]+=kv
for a in (110,400,800):
    print(a,{k:(round(R[(k,a,'starts')]/n,2),round(R[(k,a,'ends')]/n,2),round(R[(k,a,'killed')]/n,2)) for k in 'AP'},'(starts,ends,killed)/match')
