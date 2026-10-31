import sys,glob,numpy as np,collections
d='reports/bp/rlsrc/traj/'
def load(prefix):
    out=[]
    for f in sorted(glob.glob(d+prefix+'__*.npz')):
        z=np.load(f); out.append(z)
    return out
def stats(prefix,label):
    cp=[1000,1500,2000,2500,2999]
    sc=collections.defaultdict(list); K=collections.defaultdict(list); fh=[];ht=[];h100=[];n=0
    for z in load(prefix):
        for j,(s,sd) in enumerate(zip(z['seeds'],z['sides'])):
            n+=1
            sco=z['score'][j]
            # last valid
            for t in cp: sc[t].append(sco[t,sd]-sco[t,1-sd])
            cls=z['cls'][j]
            for tm,key in ((sd,'o'),(1-sd,'p')):
                ch=(cls[:,5*tm:5*tm+5]==1)
                hh=np.nonzero(ch.any(1))[0]
                (fh if key=='o' else fh).append((key,hh[0] if len(hh) else -1))
                ht.append((key,ch.sum()/25.0))
                h100.append((key,ch[100].sum()))
            k=z['kills'][z['kills'][:,0]==j]
            for _,t,kl,v,vc,kc in k:
                vt=0 if v<5 else 1
                mine_victim = (vt==sd)
                ph='open' if t<110 else ('early' if t<500 else 'late')
                K[(('lost' if mine_victim else 'got'),ph,int(vc))].append(1)
    print(f'== {label} n={n}')
    print(' score diff(ours-opp) @tick',{t:round(float(np.mean(v)),1) for t,v in sc.items()})
    for key in 'op':
        print(f' {key}: first hunter med',np.median([x for k,x in fh if k==key]),'hunter-s',round(np.mean([x for k,x in ht if k==key]),1),'hunters@100',round(np.mean([x for k,x in h100 if k==key]),2))
    for ph in ('open','early','late'):
        print(f' kills {ph}: '+' '.join(f"{w}-cls{c}:{len(K[(w,ph,c)])/n:.2f}" for w in ('got','lost') for c in (0,1,2)))
for p,l in [('bp__B3','RL v B3'),('bp__B4','RL v B4'),('bp__C0','RL v C0'),('B4__B3','B4 v B3'),('B3__B3','B3 v B3'),('B4__B4','B4 v B4')]:
    stats(p,l)
