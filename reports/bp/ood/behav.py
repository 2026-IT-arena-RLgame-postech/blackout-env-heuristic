import json,sys,numpy as np
from collections import defaultdict
from blackout_env.train.distill.ood_bp import OODBP
names={d:n for n,d,_,_ in OODBP}; names['B6']='B6'
d=defaultdict(list)
for l in open(sys.argv[1]+'/matches.jsonl'):
    r=json.loads(l); d[r['ours']].append(r)
def f(v,k,w='us'): return np.mean([x[w].get(k,0.0) for x in v])
print(f"{'name':16s} n   hun_s@20/40/60  hun_s(all) car_s  pick  steal   dep  | kills(H/C) died_col  died_hun  cargo_lost | opp: hun_s steal dep | len")
for desc,v in d.items():
    print(f"{names.get(desc,desc)[:16]:16s} {len(v):2d}  {f(v,'w20.hun_s'):4.1f}/{f(v,'w40.hun_s'):4.1f}/{f(v,'w60.hun_s'):5.1f} {f(v,'hun_s'):7.0f} {f(v,'car_s'):6.0f} {f(v,'pick_amt'):5.0f} {f(v,'steal_amt'):5.1f} {f(v,'dep_amt'):5.0f} | {f(v,'kill_by_hun'):4.1f}/{f(v,'kill_by_col'):4.1f} {f(v,'die_col'):8.1f} {f(v,'die_hun'):8.1f} {f(v,'lost_die_amt'):8.1f} | {f(v,'hun_s','them'):6.0f} {f(v,'steal_amt','them'):5.1f} {f(v,'dep_amt','them'):5.0f} | {np.mean([x['ticks'] for x in v])/50:.0f}s")
