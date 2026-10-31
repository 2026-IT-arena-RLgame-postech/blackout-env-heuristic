import glob,numpy as np,collections
d='reports/bp/rlsrc/traj/'
def run(prefix,label,t0=800):
    R=collections.defaultdict(lambda: collections.defaultdict(float))
    for f in sorted(glob.glob(d+prefix+'__*.npz')):
        z=np.load(f)
        for j,(s,sd) in enumerate(zip(z['seeds'],z['sides'])):
            alive=~np.isnan(z['pos'][j,:,0,0]); T=alive.sum()
            if T<=t0: continue
            for k,tm in (('rl',sd),('opp',1-sd)):
                sl=slice(5*tm,5*tm+5)
                ch=(z['cls'][j,t0:T,sl]==1)
                hs=ch.sum()
                R[k]['hunter_ticks']+=hs
                # moving
                p=z['pos'][j,t0:T,sl]; 
                sp=np.hypot(*(np.diff(p,axis=0).transpose(2,0,1)))
                R[k]['moving']+=((sp>0.01)&ch[1:]).sum()
                R[k]['carriers_ticks']+=(z['cls'][j,t0:T,sl]==2).sum()
                R[k]['ticks']+=T-t0
            kk=z['kills'][z['kills'][:,0]==j]
            for _,t,kl,v,vc,kc in kk:
                if t<t0 or vc!=0: continue
                vt=0 if v<5 else 1; killer_team='opp' if vt==sd else 'rl'
                hv=z['held'][j,t,v]
                R[killer_team]['kills']+=1
                R[killer_team]['kills_held']+=(hv>=0)
                own=(1,20) if v<5 else (20,1)
                R[killer_team]['kdist']+=np.hypot(*(z['pos'][j,t,v]-own))
    print('==',label)
    for k in ('rl','opp'):
        r=R[k]; print(f" {k}: hunter-share {r['hunter_ticks']/r['ticks']/5:.3f} kills/hunter-s {r['kills']/r['hunter_ticks']*25:.3f} moving {r['moving']/r['hunter_ticks']:.2f} kills_held {r['kills_held']/r['kills']:.2f} victim-dist-base {r['kdist']/r['kills']:.1f} carrier-share {r['carriers_ticks']/r['ticks']/5:.3f}")
for p,l in [('bp__B4','RL(=rl) v B4(opp)'),('bp__C0','RL v C0')]: run(p,l)
