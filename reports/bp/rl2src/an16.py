import glob,numpy as np,collections,sys
d=sys.argv[1]; pref=sys.argv[2]
C=collections.defaultdict(collections.Counter)
for f in sorted(glob.glob(d+pref+'__*.npz')):
    z=np.load(f)
    for j,(s,sd) in enumerate(zip(z['seeds'],z['sides'])):
        T=(~np.isnan(z['pos'][j,:,0,0])).sum()
        if T<400: continue
        h=lambda tm,t:(z['cls'][j,t,5*tm:5*tm+5]==1).sum()
        for t0,t1 in ((130,200),(130,300),(200,400)):
            C[(t0,t1)][((h(sd,t0),h(1-sd,t0)),(h(sd,t1),h(1-sd,t1)))]+=1
for k,c in C.items():
    print(k)
    by=collections.defaultdict(list)
    for (a,b),n in c.items(): by[a].append((b,n))
    for a in sorted(by):
        tot=sum(n for _,n in by[a]); mean=np.mean([b[0]-b[1] for b,n in by[a] for _ in range(n)])
        print('  start(RL,B4)=',a,'n',tot,'-> ',sorted(by[a],key=lambda x:-x[1])[:5],'mean diff end %.2f'%mean)
