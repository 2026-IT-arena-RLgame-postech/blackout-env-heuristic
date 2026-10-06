import sys, numpy as np
sys.path.insert(0,'examples'); sys.path.insert(0,'.')
from d1_why import load, sanct_dist, F32, COLLECTOR, HUNTER
out=sys.argv[1]
for spec in sys.argv[2].split(';'):
    ours,opp=spec.split('@'); D=load(out,ours,opp); k=D['shadows'].index('v30')
    side=D['side'].astype(int); n=len(side); us=np.stack([np.arange(5)+5*s for s in side])
    fl=D['flags'][k]; sh=D['shmv'][k].astype(F32); own=D['ownmv'].astype(F32)
    print('==',spec,'n',n)
    for a,z in ((60,75),(75,85),(85,90),(90,94),(94,99)):
        jn=((fl[a:z]&4)>0).any(-1)   # [t,n]
        st=(fl[a:z]&2)>0
        # reserve unit = staged by teacher
        tst=np.linalg.norm(sh[a:z],axis=-1)<1e-3; est=np.linalg.norm(own[a:z],axis=-1)<1e-3
        print(f'  t{a}-{z}: T-joining {jn.mean():.2f}  staged-present {st.any(-1).mean():.2f}  staged&T-stand {(st&tst).sum()/max(st.sum(),1):.2f}  staged&exec-stand {(st&est).sum()/max(st.sum(),1):.2f}  staged&T-join-move {(st&~tst&jn[...,None]).sum()/max(st.sum(),1):.2f}')
    # reserve dist distribution at 98 (4th-nearest collector) and h99
    r=[]; h=[]
    for b in range(n):
        c=D['cls'][98,b,us[b]]; d=np.sort(sanct_dist(D['pos'][98,b,us[b]].astype(F32))[c==COLLECTOR]); r.append(d[3] if len(d)>3 else np.nan)
        h.append((D['cls'][99,b,us[b]]==HUNTER).sum())
    r=np.array(r); h=np.array(h)
    print('  reserve@98 hist <0.1:',int((r<0.1).sum()),' 0.1-0.4:',int(((r>=0.1)&(r<0.4)).sum()),' 0.4-0.6:',int(((r>=0.4)&(r<0.6)).sum()),' >0.6:',int((r>=0.6).sum()),' h99 counts',np.bincount(h,minlength=5).tolist())
