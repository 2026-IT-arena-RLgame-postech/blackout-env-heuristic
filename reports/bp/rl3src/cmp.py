"""paired margin vs the first (base) descriptor, per (seed, side): python cmp.py matches.jsonl [base_substr]"""
import json,sys,numpy as np,collections
rows=[json.loads(l) for l in open(sys.argv[1])]
g=collections.defaultdict(dict)
for r in rows: g[r['ours']][(r['seed'],r['side'])]=r
def short(d):
    return d.split('race_order=true')[-1][:90] if 'race_order' in d else d[:60]
base=sys.argv[2] if len(sys.argv)>2 else None
keys=list(g)
b=[k for k in keys if (base is None and k.endswith('sentry_offset=6')) or (base and k==base)] or keys[:1]
b=b[0]
def se(x): return np.std(x,ddof=1)/np.sqrt(len(x)) if len(x)>1 else float('nan')
for k in keys:
    d=g[k]; m=[r['margin'] for r in d.values()]
    out=f"{short(k):95s} n={len(d)} raw {np.mean(m):+6.1f}±{se(m):4.1f} win {np.mean(np.array(m)>0):.2f}"
    for s in (0,1):
        ms=[r['margin'] for (sd,si),r in d.items() if si==s]; out+=f" s{s} {np.mean(ms):+5.1f}"
    if k!=b:
        pk=[(g[k][x]['margin']-g[b][x]['margin']) for x in d if x in g[b]]
        # average over sides per seed
        by=collections.defaultdict(list)
        for x in d:
            if x in g[b]: by[x[0]].append(g[k][x]['margin']-g[b][x]['margin'])
        pv=[np.mean(v) for v in by.values()]
        out+=f" | paired vs base {np.mean(pv):+5.1f}±{se(pv):.1f}"
    print(out)
