import json,sys,numpy as np
from collections import defaultdict
from blackout_env.train.distill.exploit_bp import NAMES
d=defaultdict(list)
for f in sys.argv[1:]:
    for l in open(f+'/matches.jsonl'):
        r=json.loads(l); d[r['ours']].append(r)
for o,v in sorted(d.items(), key=lambda kv: NAMES.get(kv[0],'zz')):
    m=np.array([r['margin'] for r in v]); sd=np.array([r['side'] for r in v])
    print(f"{NAMES.get(o,'?'):10s} n={len(m):3d} {m.mean():+6.1f}±{m.std(ddof=1)/len(m)**.5:4.1f} win {(m>0).mean():.2f} A{m[sd==0].mean():+5.0f} B{m[sd==1].mean():+5.0f}")
