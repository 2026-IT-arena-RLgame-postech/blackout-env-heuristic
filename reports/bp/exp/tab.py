import json,sys,numpy as np
from collections import defaultdict
d=defaultdict(list)
for f in sys.argv[1:]:
    for l in open(f+'/matches.jsonl'):
        r=json.loads(l); d[r['ours']].append(r)
rows=[]
for o,v in d.items():
    m=np.array([r['margin'] for r in v]); sd=np.array([r['side'] for r in v])
    rows.append((m.mean(),o,len(m),m.std(ddof=1)/len(m)**.5,(m>0).mean(),m[sd==0].mean(),m[sd==1].mean()))
for mm,o,n,se,w,a,b in sorted(rows,reverse=True):
    print(f"{mm:+6.1f}±{se:4.1f} n={n:3d} win {w:.2f} A{a:+5.0f} B{b:+5.0f}  {o.replace('B6,','')[:140]}")
