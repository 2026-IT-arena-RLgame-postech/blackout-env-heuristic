import sys,numpy as np
sys.path.insert(0,'scripts'); sys.path.insert(0,'.')
import bp_loop as L
from blackout_env.heuristics.params import resolve_descriptor, split_descriptor
gen,basen,opps=sys.argv[1],sys.argv[2],sys.argv[3].split(";")
t=L.table(L.load(gen))
base=basen; bd=split_descriptor(base)[1]
res=[]
for c in t:
    if c==base: continue
    cd=split_descriptor(c)[1]
    diff={k:v for k,v in cd.items() if bd.get(k)!=v}
    d=L.paired(t,c,base,opps)
    res.append((d[0],d[1],str(diff)[:200]))
for r in sorted(res,reverse=True): print(f"{r[0]:+.1f}±{r[1]:.1f} {r[2]}")
