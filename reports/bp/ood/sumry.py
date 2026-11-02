import json,sys,numpy as np
from collections import defaultdict
d=defaultdict(list)
for f in sys.argv[1:]:
    for l in open(f+'/matches.jsonl'):
        r=json.loads(l); d[(r['ours'],r['opponent'])].append(r)
for (o,p),v in d.items():
    m=np.array([r['margin'] for r in v]); 
    print(f"{o[:70]:70s} vs {p[:8]:8s} n={len(m):3d} {m.mean():+7.1f}±{m.std(ddof=1)/len(m)**.5:4.1f} win {(m>0).mean():.2f} H-s {np.mean([r['hunter_s'] for r in v]):4.0f}/{np.mean([r['opp_hunter_s'] for r in v]):4.0f} first {np.mean([r['first_ours'] for r in v if r['first_ours']>=0] or [-1]):.0f} none {sum(r['first_ours']<0 for r in v)} ticks {np.mean([r['ticks'] for r in v]):.0f}")
