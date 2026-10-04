import numpy as np, sys, json
from collections import Counter
from examples.late_search_why import load_tag, match_facts, root_features
from pathlib import Path
out=Path('reports/late_search_why')
opp=sys.argv[1]
S=load_tag(out,'search',opp); P=load_tag(out,'plain',opp)
fS=match_facts(S); fP=match_facts(P)
br=json.loads(str(S['branches']))
rf=root_features(S)
def sd(p):
    dx=np.maximum(np.maximum(11-p[...,0],p[...,0]-13),0); dy=np.maximum(np.maximum(11-p[...,1],p[...,1]-13),0); return np.hypot(dx,dy)
g=rf['game']; t=rf['tick']; ch=rf['choice']
tied=(fS['race']==0)
for lo,hi in ((0,50),(50,100),(100,150),(150,250),(250,500),(500,4000)):
    m=(t>=lo)&(t<hi)&(ch>0)
    c=Counter()
    for i in np.nonzero(m)[0]:
        name=br[ch[i]]
        if name.startswith('samp'): c['samp']+=1; continue
        u=int(name[3]); side=int(S['side'][g[i]]); gu=5*side+u
        res = sd(S['pos'][98,g[i],gu])>1.0
        c[('reserve' if res else 'walker') + ('/H' if S['cls'][t[i],g[i],gu]==1 else '')]+=1
    tot=((t>=lo)&(t<hi)).sum()
    print(f"{lo}-{hi}: roots {tot} dev {m.sum()} ({m.sum()/max(tot,1):.2f})", dict(c))
