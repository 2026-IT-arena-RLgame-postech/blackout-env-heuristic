import sys,json,collections,numpy as np
sys.path.insert(0,'scripts'); sys.path.insert(0,'.')
import bp_loop as L
from blackout_env.heuristics.params import split_descriptor
gen,basen=sys.argv[1],sys.argv[2]
bd=split_descriptor(basen)[1]
acc=collections.defaultdict(list)
for r in L.load(gen):
    cd=split_descriptor(r['ours'])[1]; diff=str({k:v for k,v in cd.items() if bd.get(k)!=v})[:60]
    acc[(diff,r['opponent'][:8] if len(r['opponent'])<10 else 'long',r['side'])].append(r['margin'])
for k in sorted(acc): print(k,round(np.mean(acc[k]),1),len(acc[k]))
