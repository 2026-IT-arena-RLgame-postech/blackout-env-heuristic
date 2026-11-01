import sys; sys.path.insert(0,'scripts'); sys.path.insert(0,'.')
import bp_loop as L, numpy as np
gen=sys.argv[1]; base=sys.argv[2]; opps=sys.argv[3].split(';')
rows=L.load(gen); t=L.table(rows)
res=[]
for c in t:
    if c==base: continue
    per=[]
    for o in opps:
        p=L.paired(t,c,base,[o]); per.append(p)
    res.append((L.paired(t,c,base,opps),per,c))
for p,per,c in sorted(res,key=lambda x:-np.nan_to_num(x[0][0],nan=-99)):
    print(f"{p[0]:+5.1f}±{p[1]:.1f} | "+' '.join(f"{q[0]:+5.1f}±{q[1]:.1f}" for q in per)+f" | raw {p[2]:+5.1f} | {c[-70:]}")
