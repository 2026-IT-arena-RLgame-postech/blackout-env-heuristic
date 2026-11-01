import glob,numpy as np,collections,sys
d=sys.argv[1]; pref=sys.argv[2]
ev=collections.defaultdict(list)
for f in sorted(glob.glob(d+pref+'__*.npz')):
    z=np.load(f)
    for j,(s,sd) in enumerate(zip(z['seeds'],z['sides'])):
        T=(~np.isnan(z['pos'][j,:,0,0])).sum()
        for k,tm in (('A',sd),('P',1-sd)):
            sl=slice(5*tm,5*tm+5); sle=slice(5*(1-tm),5*(1-tm)+5)
            c=z['cls'][j,:T,sl]; ce=z['cls'][j,:T,sle]; p=z['pos'][j,:T,sl]; pe=z['pos'][j,:T,sle]
            for t in range(101,T):
                for u in range(5):
                    if c[t,u]==1 and c[t-1,u]!=1:
                        nh=(c[t-1]==1).sum(); ne=(ce[t-1]==1).sum()
                        if nh>=2: continue
                        q=p[t-5,u]; ds=np.hypot(q[0]-12,q[1]-12)
                        eh=np.nonzero(ce[t-1]==1)[0]
                        de=min([np.hypot(*(pe[t-5,i]-q)) for i in eh]) if len(eh) else np.nan
                        deS=min([np.hypot(pe[t-1,i,0]-12,pe[t-1,i,1]-12) for i in eh]) if len(eh) else np.nan
                        oh=np.nonzero(c[t-1]==1)[0]
                        dsh=min([np.hypot(p[t-1,i,0]-12,p[t-1,i,1]-12) for i in oh]) if len(oh) else np.nan
                        ev[(k,nh,ne)].append((t,ds,de,deS,dsh))
for key in sorted(ev):
    a=np.array(ev[key]); 
    if len(a)<4: continue
    print(pref,key,'n',len(a),'t med %.0f'%np.median(a[:,0]),'unit dist sanct (5 ticks before) med %.1f'%np.median(a[:,1]),'to nearest enemy H %.1f'%np.nanmedian(a[:,2]),'enemy H dist sanct %.1f'%np.nanmedian(a[:,3]),'own H dist sanct %.1f'%np.nanmedian(a[:,4]))
