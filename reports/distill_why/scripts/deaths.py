import sys, numpy as np
sys.path.insert(0,'examples'); sys.path.insert(0,'.')
from d1_why import load
from blackout_env.sim import core
out=sys.argv[1]; opp=sys.argv[2]
names='CHK'
for ours in sys.argv[3].split(';'):
    D=load(out,ours,opp); side=D['side'].astype(int); n=len(side); ev=D['events']
    c={}
    for row in ev:
        b,t,kind,killer,victim,vcls,kcls=map(int,row)
        if kind!=core.EV_KILL: continue
        s=t/25
        w='5-10' if 5<=s<10 else '10-20' if 10<=s<20 else '20-30' if 20<=s<30 else None
        if w is None: continue
        ours_v = (victim//5)==side[b]
        key=(w,'our' if ours_v else 'opp',names[vcls],'by'+(names[kcls] if killer>=0 else '-'))
        c[key]=c.get(key,0)+1
    print('==',ours, ' '.join(f"{'/'.join(k)}:{v/n:.2f}" for k,v in sorted(c.items()) if v/n>=0.1))
