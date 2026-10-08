import sys, numpy as np
sys.path.insert(0,'examples'); sys.path.insert(0,'.')
from d1_why import load, outcome_rows, sanct_dist, HUNTER, COLLECTOR, F32
from blackout_env.sim import core
out=sys.argv[1]; opp=sys.argv[2]
for o in sys.argv[3].split(';'):
    D=load(out,o,opp); F=outcome_rows(D); side=D['side'].astype(int); n=len(side); ev=D['events']
    us=[set(range(5*s,5*s+5)) for s in side]
    lost=F['margin']<-20; h4=F['h99']==4
    res={}
    for hv,sel in (('3',~h4),('4',h4)):
        nxt_o=[];nxt_p=[];kills_near=[]
        for b in np.nonzero(sel)[0]:
            e=ev[(ev[:,0]==b)&(ev[:,2]==core.EV_TRANSFORM)&(ev[:,4]==HUNTER)&(ev[:,1]>104)]
            mo=[t for t,a in zip(e[:,1],e[:,3]) if a in us[b]]; mp=[t for t,a in zip(e[:,1],e[:,3]) if a not in us[b]]
            nxt_o.append(min(mo) if mo else 9999); nxt_p.append(min(mp) if mp else 9999)
            k=ev[(ev[:,0]==b)&(ev[:,2]==core.EV_KILL)&(ev[:,1]>100)&(ev[:,1]<250)]
            c=0
            for row in k:
                t,killer,victim,vcls=row[1],row[3],row[4],row[5]
                if killer in us[b] and vcls==COLLECTOR:
                    if sanct_dist(D['pos'][t-1,b,victim].astype(F32))<7: c+=1
            kills_near.append(c)
        nxt_o=np.array(nxt_o); nxt_p=np.array(nxt_p); kn=np.array(kills_near)
        first=nxt_o<nxt_p; L=lost[sel]
        res[hv]=f"h99={hv} n={sel.sum():2d}: we get next Hunter first {first.mean():.2f} (loss|first {L[first].mean() if first.any() else np.nan:.2f}, loss|not {L[~first].mean() if (~first).any() else np.nan:.2f}); our kills of enemy Collectors <7 tiles of sanctuary 4-10s {kn.mean():.2f}/game"
    print(f"{o:7s}", ' || '.join(res.values()))
