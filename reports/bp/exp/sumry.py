import json,sys,numpy as np
from collections import defaultdict
from blackout_env.train.distill.ood_bp import NAMES
from blackout_env.heuristics.params import resolve_descriptor
nm=dict(NAMES); 
for k in ("B6",): nm[resolve_descriptor(k)]=k
try:
    import blackout_env.train.distill.exploit_bp as X
    for n,d,_ in X.EXPLOITERS: nm[d]=n
except Exception: pass
d=defaultdict(list)
for f in sys.argv[1:]:
    for l in open(f+'/matches.jsonl'):
        r=json.loads(l); d[(r['ours'],r['opponent'])].append(r)
for (o,p),v in sorted(d.items()):
    m=np.array([r['margin'] for r in v]); sd=np.array([r['side'] for r in v])
    s=lambda x:nm.get(x,x[-28:] if x.startswith('net:') else x[:30])
    print(f"{s(o)[:24]:24s} vs {s(p)[:16]:16s} n={len(m):3d} {m.mean():+7.1f}±{m.std(ddof=1)/len(m)**.5:4.1f} win {(m>0).mean():.2f} sideA {m[sd==0].mean():+.0f} sideB {m[sd==1].mean():+.0f} H-s {np.mean([r['hunter_s'] for r in v]):4.0f}/{np.mean([r['opp_hunter_s'] for r in v]):4.0f} ticks {np.mean([r['ticks'] for r in v]):.0f}")
