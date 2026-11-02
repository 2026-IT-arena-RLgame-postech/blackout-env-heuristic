import glob,sys,numpy as np
from collections import Counter
d=sys.argv[1]
bins=[(0,100),(100,200),(200,400),(400,800),(800,1600),(1600,4000)]
H={b:[[],[]] for b in bins}; S={b:[] for b in bins}
kc=Counter(); loc=Counter(); n=0; fin=[]
firstH=[[],[]]
for f in sorted(glob.glob(d+'/*.npz')):
    z=np.load(f)
    for j,sd in enumerate(z['sides']):
        n+=1
        cls=z['cls'][j]; T=int((cls[:,0]>=0).sum())
        us=slice(5*sd,5*sd+5); th=slice(5*(1-sd),5*(1-sd)+5)
        hu=(cls[:T,us]==1).sum(1); ho=(cls[:T,th]==1).sum(1)
        for b in bins:
            a,e=b[0],min(b[1],T)
            if e>a: H[b][0].append(hu[a:e].mean()); H[b][1].append(ho[a:e].mean())
        for k,h in enumerate((hu,ho)):
            w=np.nonzero(h>0)[0]; firstH[k].append(w[0] if len(w) else -1)
        sc=z['score'][j,T-1]; fin.append(int(sc[sd])-int(sc[1-sd]))
        pos=z['pos'][j]
        for _,t,a,v,vc,kcl in z['kills'][z['kills'][:,0]==j]:
            vt=v//5; us_v = (vt==sd)
            kc[('RLvictim' if us_v else 'OPPvictim', 'hunterV' if vc==1 else ('carrierV' if vc==2 else 'collV'), 'byH' if kcl==1 else 'byOther')]+=1
            x,y=pos[t,v]
            if vt==1: x,y=y,x   # into team0 frame (victim's own frame)
            if us_v and vc!=1:
                zone='ownbase' if (x<5 and y>=19) else ('sancR<8' if np.hypot(x-12,y-12)<0 else 'other')
                loc[(('RL','coll/car'), int(x//4*4), int(y//4*4))]+=1
print(d,'n',n,'final margin',np.mean(fin),'win',np.mean(np.array(fin)>0))
print('first hunter tick RL/opp',np.mean([x for x in firstH[0] if x>=0]),np.mean([x for x in firstH[1] if x>=0]))
for b in bins: print(b,'RL hunters %.2f opp hunters %.2f'%(np.mean(H[b][0]),np.mean(H[b][1])))
for k,v in sorted(kc.items()): print(k,round(v/n,2))
