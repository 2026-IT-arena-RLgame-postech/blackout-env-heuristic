import json,sys,numpy as np,collections
rows=[json.loads(l) for l in open(sys.argv[1]+'/matches.jsonl')]
g=collections.defaultdict(list)
for r in rows: g[(r['ours'][-12:],r['opponent'],r['side'])].append(r)
def se(x): return np.std(x,ddof=1)/np.sqrt(len(x))
for (u,o,s),v in sorted(g.items()):
    m=[r['margin'] for r in v]
    f=np.array([r['first_ours'] for r in v]); fo=np.array([r['first_opp'] for r in v])
    print(f"{u[-6:]:6s} vs {o[:8]:8s} side{s} n={len(v)} margin {np.mean(m):+6.1f}±{se(m):4.1f} win {np.mean(np.array(m)>0):.2f} first med {np.median(f)}/{np.median(fo)} H-s {np.mean([r['hunter_s'] for r in v]):.0f}/{np.mean([r['opp_hunter_s'] for r in v]):.0f} h99 {np.mean([r['h99_ours'] for r in v]):.2f}/{np.mean([r['h99_opp'] for r in v]):.2f} score {np.mean([r['score_ours'] for r in v]):.0f}/{np.mean([r['score_opp'] for r in v]):.0f}")
