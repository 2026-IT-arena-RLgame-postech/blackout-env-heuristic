import json, sys, numpy as np
from collections import defaultdict
path, base_name = sys.argv[1], sys.argv[2]
rows=[json.loads(l) for l in open(path)]
by=defaultdict(dict)
for r in rows: by[(r['ours'],r['opponent'])][(r['seed'],r['side'])]=r
def se(x): x=np.asarray(x,float); return x.std(ddof=1)/np.sqrt(len(x))
def wtl(v,a,b):
    x=np.array([r[a] if r[a]>=0 else 1e9 for r in v]); y=np.array([r[b] if r[b]>=0 else 1e9 for r in v]); s=np.sign(y-x)
    return f"{(s>0).sum()}/{(s==0).sum()}/{(s<0).sum()}"
short={'p17':'P17','strategic_v28:reserve_stage=0.5,trade_hold=advance':'V28','p16':'P16','strategic_v27:reserve_stage=1.5':'V27(1.5)','strategic_v25':'V25','p14b':'P14b'}
cands=sys.argv[3].split(',')
print("| 후보 | 상대 | 경기 | 점수차 | **Δ vs P17** | 두 번째 경주 승/동/패 | 첫 Hunter 99 | 예비 res100 중앙값 (p10) |")
print("|---|---|---|---|---|---|---|---|")
pooled=defaultdict(lambda: defaultdict(list))
for c in cands:
  for o in short:
    d=by.get((c,o))
    if not d: continue
    base=by[(base_name,o)]
    ms=[r['margin'] for r in d.values()]
    per=defaultdict(list)
    for k in d:
        if k in base: per[k[0]].append(d[k]['margin']-base[k]['margin']); pooled[c][k[0]].append(d[k]['margin']-base[k]['margin'])
    ps=[np.mean(v) for v in per.values()]
    f=[r['first_ours'] for r in d.values()]
    res=[r['reserve100'] for r in d.values()]
    dl=f"{np.mean(ps):+.1f}±{se(ps):.1f} ({len(ps)}시드)" if c!=base_name else '–'
    print(f"| {c} | {short[o]} | {len(d)} | {np.mean(ms):+.1f}±{se(ms):.1f} | {dl} | {wtl(list(d.values()),'second_ours','second_opp')} | {sum(x==99 for x in f)}/{len(f)} | {np.nanmedian(res):.1f} ({np.nanquantile(res,0.1):.1f}) |")
for c in cands:
  if c==base_name: continue
  ps=[np.mean(v) for v in pooled[c].values()]
  print(c,'pooled all opponents', f"{np.mean(ps):+.2f}±{se(ps):.2f}")
for c in cands:
  d=by.get((c,'p17'))
  if d:
    t=np.array([r['ticks'] for r in d.values()])/25; print(c,'vs p17 len mean %.0f p90 %.0f'%(t.mean(),np.percentile(t,90)))
