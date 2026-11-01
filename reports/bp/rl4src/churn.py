import glob,numpy as np,collections,sys
d='reports/bp/rl4/traj/'
pre=sys.argv[1]
C=collections.Counter(); gaps=collections.Counter(); first=collections.Counter(); n=0
for f in sorted(glob.glob(d+pre+'__*.npz')):
    z=np.load(f)
    for j,sd in enumerate(z['sides']):
        cls=z['cls'][j]; T=(~np.isnan(z['pos'][j,:,0,0])).sum(); n+=1
        ev=[]  # (t, unit)
        for u in range(10):
            c=cls[:T,u]
            for t in np.nonzero((c[1:]==1)&(c[:-1]!=1))[0]+1:
                if t>105: ev.append((t,u))
        ev.sort()
        for t,u in ev:
            life=0
            while t+life<T and cls[t+life,u]==1: life+=1
            tm=u//5
            # other team transform within +-3 ticks?
            oth=[(t2,u2) for t2,u2 in ev if u2//5!=tm and abs(t2-t)<=3]
            C[(life<=5, bool(oth))]+=1
            if life<=5 and oth:
                gaps[min(oth,key=lambda e:abs(e[0]-t))[0]-t]+=1
            if life<=5 and not oth:
                # killed by an existing enemy hunter? count enemy hunters alive at t
                eh=(cls[t,5*(1-tm):5*(1-tm)+5]==1).sum()
                C[('noother_enemyH',int(eh))]+=1
print(pre,n,'events (life<=5, simultaneous other-team transform):',{k:v for k,v in C.items() if isinstance(k[0],bool)})
print(' lonely short-lived transforms by enemy hunters alive:',{k[1]:v for k,v in C.items() if k[0]=='noother_enemyH'})
print(' gap (other-this ticks) for short-lived simultaneous:',dict(sorted(gaps.items())))
