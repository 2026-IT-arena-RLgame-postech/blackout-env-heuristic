import glob,numpy as np,collections,sys
d='reports/bp/rlsrc/traj/'
def run(prefix,label):
    cps=[100,110,130,160,200,300,400,600]
    H={'o':collections.defaultdict(list),'p':collections.defaultdict(list)}
    zones=collections.defaultdict(list); n=0
    for f in sorted(glob.glob(d+prefix+'__*.npz')):
        z=np.load(f)
        for j,(s,sd) in enumerate(zip(z['seeds'],z['sides'])):
            n+=1; alive=~np.isnan(z['pos'][j,:,0,0]); T=alive.sum()
            for k,tm in (('o',sd),('p',1-sd)):
                for t in cps:
                    if t<T: H[k][t].append((z['cls'][j,t,5*tm:5*tm+5]==1).sum())
            kk=z['kills'][z['kills'][:,0]==j]
            for _,t,kl,v,vc,kc in kk:
                if 110<=t<800 and vc==0:
                    mine=((0 if v<5 else 1)==sd)
                    p=z['pos'][j,t,v]
                    # base corner: team0 spawn (1,20)? dist to own sanctuary
                    own=(1,20) if v<5 else (20,1)
                    dist=np.hypot(p[0]-own[0],p[1]-own[1])
                    sanct=np.hypot(max(11-p[0],0,p[0]-13),max(11-p[1],0,p[1]-13))
                    zones[('lost' if mine else 'got')].append((dist,sanct,t,kc))
    print('==',label,n)
    for k in 'op': print(' hunters',k,{t:round(float(np.mean(H[k][t])),2) for t in cps})
    for w in ('got','lost'):
        a=np.array(zones[w]); 
        if len(a): print(f' {w} collector kills 110-800: n/match {len(a)/n:.1f} median dist-to-victim-base {np.median(a[:,0]):.1f}, frac<6 {np.mean(a[:,0]<6):.2f}, frac near sanct(<4) {np.mean(a[:,1]<4):.2f}, killer hunter {np.mean(a[:,3]==1):.2f}')
for p,l in [('bp__B3','RL v B3'),('B4__B3','B4 v B3'),('bp__B4','RL v B4'),('B4__B4','B4 v B4')]: run(p,l)
