import json,sys,numpy as np,collections
rows=[json.loads(l) for l in open(sys.argv[1]+'/matches.jsonl')]
g=collections.defaultdict(dict)
for r in rows: g[(r['ours'][-12:] if 'net:' in r['ours'] else r['ours'][:20],r['opponent'][:20])][(r['seed'],r['side'])]=r
for (u,o),d in sorted(g.items()):
    out=[]
    for s in (0,1):
        m=[r['margin'] for (sd,sdd),r in d.items() if sdd==s]
        out.append(f"side{s} {np.mean(m):+6.1f}±{np.std(m,ddof=1)/len(m)**.5:.1f}")
    seeds=sorted({k[0] for k in d})
    avg=[np.mean([d[(sd,s)]['margin'] for s in (0,1) if (sd,s) in d]) for sd in seeds]
    # second race
    w=t=l=0
    for r in d.values():
        a,b=r['second_ours'],r['second_opp']
        if a is None or b is None or a<0 or b<0: continue
        w+=a<b; l+=a>b; t+=a==b
    print(f"{u:22s} vs {o:10s} n={len(d)} {out[0]} {out[1]} avg {np.mean(avg):+.1f}±{np.std(avg,ddof=1)/len(avg)**.5:.1f} win {np.mean([r['margin']>0 for r in d.values()]):.2f} 2ndrace w/t/l {w}/{t}/{l}")
