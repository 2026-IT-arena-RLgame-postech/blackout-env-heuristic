import glob,numpy as np,collections,sys
d='reports/bp/rl3/traj/'
def run(prefix,label,ph=((110,400),(400,800),(800,1500))):
    R=collections.defaultdict(lambda: collections.defaultdict(float)); n=0
    for f in sorted(glob.glob(d+prefix+'__*.npz')):
        z=np.load(f)
        for j,(s,sd) in enumerate(zip(z['seeds'],z['sides'])):
            n+=1; T=(~np.isnan(z['pos'][j,:,0,0])).sum()
            pos=z['pos'][j]; cls=z['cls'][j]; held=z['held'][j]
            for k,tm in (('A',sd),('P',1-sd)):  # A = ours(first label), P = opp
                sl=slice(5*tm,5*tm+5); en=slice(5*(1-tm),5*(1-tm)+5)
                ebase=(20,1) if tm==0 else (1,20)  # enemy base location
                obase=(1,20) if tm==0 else (20,1)
                for a,b in ph:
                    b2=min(b,T)
                    if b2<=a: continue
                    c=cls[a:b2,sl]; p=pos[a:b2,sl]
                    h=(c==1)
                    key=(k,a)
                    R[key]['ticks']+=b2-a
                    R[key]['hunters']+=h.sum()
                    R[key]['h2']+=((h.sum(1))>=2).sum()
                    R[key]['h0']+=((h.sum(1))==0).sum()
                    R[key]['carriers']+=(c==2).sum()
                    R[key]['coll']+=(c==0).sum()
                    # hunter dist to enemy base
                    dd=np.hypot(p[...,0]-ebase[0],p[...,1]-ebase[1])
                    R[key]['h_near_ebase']+=(h&(dd<7)).sum()
                    do=np.hypot(p[...,0]-obase[0],p[...,1]-obase[1])
                    R[key]['h_near_obase']+=(h&(do<7)).sum()
                    R[key]['coll_out_near_ebase']+=((c==0)&(dd<7)).sum()
                    sanct=np.hypot(np.maximum(11-p[...,0],0)+np.maximum(p[...,0]-13,0),np.maximum(11-p[...,1],0)+np.maximum(p[...,1]-13,0))
                    R[key]['h_near_sanct']+=(h&(sanct<3)).sum()
                    # collectors away from own base (dist>7)
                    R[key]['coll_far']+=((c==0)&(do>7)).sum()
                    R[key]['carr_far']+=((c==2)&(do>7)).sum()
                    R[key]['holding']+=((held[a:b2,sl]>=0)).sum()
            kk=z['kills'][z['kills'][:,0]==j]
            for _,t,kl,v,vc,kc in kk:
                vt=0 if v<5 else 1
                who='P' if vt==sd else 'A'  # who is the killer: A if victim is opp
                who='A' if vt!=sd else 'P'
                for a,b in ph:
                    if a<=t<b and vc==0:
                        R[(who,a)]['kills']+=1
                        R[(who,a)]['kills_held']+=(held[t,v]>=0)
    print('==',label,n)
    for a,b in ph:
        for k in 'AP':
            r=R[(k,a)]
            tt=r['ticks']; 
            print(f" {a}-{b} {k}: hunters/tick {r['hunters']/tt:.2f} h2 {r['h2']/tt*1:.2f} h0 {r['h0']/tt:.2f} carr {r['carriers']/tt:.2f} coll-far {r['coll_far']/tt:.2f} carr-far {r['carr_far']/tt:.2f} H@ebase {r['h_near_ebase']/max(r['hunters'],1):.2f} H@own {r['h_near_obase']/max(r['hunters'],1):.2f} H@sanct {r['h_near_sanct']/max(r['hunters'],1):.2f} kills/match {r['kills']/n:.1f} kills/Hs {r['kills']/max(r['hunters'],1)*25:.2f} killsheld {r['kills_held']/max(r['kills'],1):.2f} hold {r['holding']/tt:.2f}")
for p,l in [('bp__B5','RL2(A) v B4(P)'),('B5__B5','B4 v B4'),('bp__C0','RL2(A) v C0(P)'),('B4__C0','B4(A) v C0(P)')]: run(p,l)
