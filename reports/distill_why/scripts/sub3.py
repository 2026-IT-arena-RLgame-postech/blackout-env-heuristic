import sys, numpy as np
sys.path.insert(0,'examples'); sys.path.insert(0,'.')
import d1_why as d
from d1_why import load, HUNTER, F32
out=sys.argv[1]
D=load(out,'v30','v28'); side=D['side'].astype(int); n=len(side)
us=np.stack([np.arange(5)+5*s for s in side])
h99=np.array([(D['cls'][99,b,us[b]]==HUNTER).sum() for b in range(n)])
d.PHASES=((4,6),(6,10),(10,20),(20,30))
for hv in (3,4):
    g=np.nonzero(h99==hv)[0]
    S={k:(v[g] if k in ('seed','side','final','end') else v) for k,v in D.items() if k not in ('events','shadows')}
    for k in ('pos','cls','htyp','score','ownmv'): S[k]=D[k][:,g]
    S['shmv']=D['shmv'][:,:,g]; S['flags']=D['flags'][:,:,g]; S['shadows']=D['shadows']
    fl=S['flags'][0]
    for k,s in enumerate(S['shadows']):
        if not d.is_net(s): continue
        t=d.agree_table(S,S['shmv'][k].astype(F32),S['ownmv'].astype(F32),fl,f'== {s} on V30 states, games with {hv} Hunters at 99 (n={len(g)})')
        print('\n'.join(l for l in t.split('\n') if ('ALL' in l or 'H-' in l or 'C-empty' in l or '==' in l or 'phase' in l)))
