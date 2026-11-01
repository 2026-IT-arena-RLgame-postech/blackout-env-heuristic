"""50 Hz trajectory summary (t = decision index, 50/s). usage: an.py D PREFIX_A [PREFIX_B ...]  (A = first-named policy)"""
import glob,numpy as np,collections,sys
d=sys.argv[1]; PS=50
ph=[(0,220),(220,800),(800,1600),(1600,3000),(3000,6000)]   # 25 Hz notes: 110-400 -> 220-800 etc.
def run(prefix):
    R=collections.defaultdict(float); n=0; dm=collections.defaultdict(list)
    st=collections.defaultdict(lambda:collections.defaultdict(float)); T_=[]
    for f in sorted(glob.glob(d+prefix+'__*.npz')):
        z=np.load(f)
        for j,(s,sd) in enumerate(zip(z['seeds'],z['sides'])):
            n+=1; T=int((~np.isnan(z['pos'][j,:,0,0])).sum()); T_.append(T)
            sc=z['score'][j][:T].astype(int); m=np.concatenate([[0],sc[:,sd]-sc[:,1-sd]])
            pos=z['pos'][j]; cls=z['cls'][j]; held=z['held'][j]
            ho=(cls[:T,5*sd:5*sd+5]==1).sum(1); hp=(cls[:T,5*(1-sd):5*(1-sd)+5]==1).sum(1)
            for a,b in ph:
                a2=min(a,T);b2=min(b,T); dm[(a,b)].append(m[b2]-m[a2])
                if b2>a2:
                    for k,tm,oh in (('A',sd,ho),('P',1-sd,hp)):
                        sl=slice(5*tm,5*tm+5); c=cls[a2:b2,sl]; p=pos[a2:b2,sl]
                        ebase=(20,1) if tm==0 else (1,20); obase=(1,20) if tm==0 else (20,1)
                        h=c==1; r=st[(k,a)]
                        r['t']+=b2-a2; r['H']+=h.sum(); r['coll']+=(c==0).sum(); r['car']+=(c==2).sum()
                        dd=np.hypot(p[...,0]-ebase[0],p[...,1]-ebase[1]); do=np.hypot(p[...,0]-obase[0],p[...,1]-obase[1])
                        r['Hebase']+=(h&(dd<7)).sum(); r['Hown']+=(h&(do<7)).sum()
                        r['collfar']+=((c==0)&(do>7)).sum(); r['hold']+=(held[a2:b2,sl]>=0).sum()
                        r['collin']+=((c==0)&(do<=4)).sum()
                    for x in (ho[a2:b2]*3+hp[a2:b2]): pass
                    for so,sp in ((0,0),(1,0),(0,1),(1,1),(2,0),(2,1),(1,2),(0,2),(2,2)):
                        st[('S',a)][(so,sp)]+=((ho[a2:b2]==so)&(hp[a2:b2]==sp)).sum()
            kk=z['kills'][z['kills'][:,0]==j]
            for _,t,kl,v,vc,kc in kk:
                if vc!=0: continue
                who='A' if (0 if v<5 else 1)!=sd else 'P'
                for a,b in ph:
                    if a<=t<b: R[(who,a)]+=1
    print(f'== {prefix} n={n} meanT={np.mean(T_)/PS:.0f}s')
    print('  margin gain/phase:',{f"{a/PS:.0f}-{b/PS:.0f}s":round(float(np.mean(dm[(a,b)])),1) for a,b in ph},'total',round(sum(float(np.mean(dm[p])) for p in ph),1))
    for a,b in ph:
        s=[]
        for k in 'AP':
            r=st[(k,a)]; t=max(r['t'],1)
            s.append(f"{k}: H {r['H']/t:.2f} coll {r['coll']/t:.2f} car {r['car']/t:.2f} Hebase {r['Hebase']/max(r['H'],1):.2f} Hown {r['Hown']/max(r['H'],1):.2f} collfar {r['collfar']/t:.2f} collin {r['collin']/t:.2f} hold {r['hold']/t:.2f} collkills {R[(k,a)]/n:.1f}")
        S=st[('S',a)]; tt=max(sum(S.values()),1)
        print(f"  {a/PS:.0f}-{b/PS:.0f}s "+' | '.join(s)+' | states(A,P) '+' '.join(f"{k[0]}{k[1]}:{v/tt:.2f}" for k,v in sorted(S.items()) if v/tt>=0.03))
for p in sys.argv[2:]: run(p)
