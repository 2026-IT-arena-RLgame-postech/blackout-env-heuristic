import sys, numpy as np
sys.path.insert(0,'examples'); sys.path.insert(0,'.')
from d1_why import load, sanct_dist, moves_agree, F32, COLLECTOR, HUNTER
out=sys.argv[1]
for spec in sys.argv[2].split(';'):
    ours,opp=spec.split('@'); D=load(out,ours,opp); k=D['shadows'].index('v30')
    side=D['side'].astype(int); n=len(side); us=np.stack([np.arange(5)+5*s for s in side])
    fl=D['flags'][k]; sh=D['shmv'][k].astype(F32); own=D['ownmv'].astype(F32)
    print('==',spec)
    # per game: the staged unit at tick 80; trace 80-99: dist, exec angle err vs teacher, exec speed
    bad=[];
    for b in range(n):
        st=np.nonzero((fl[80,b]&2)>0)[0]
        if not len(st): continue
        u=st[0]
        d=sanct_dist(D['pos'][80:100,b,us[b][u]].astype(F32))
        ok,sa,sb,ang=moves_agree(own[80:99,b,u],sh[80:99,b,u])
        h4=(D['cls'][99,b,us[b]]==HUNTER).sum()==4
        bad.append((h4, (~ok).sum(), np.round(d[[0,8,14,18]],2).tolist(), np.round(ang[~ok],0).tolist()[:6], int(sb.sum()), int(sa.sum())))
    for h4 in (True,False):
        g=[x for x in bad if x[0]==h4]
        print(f'  h99=4 {h4}: n={len(g)} mean disagreeing ticks 80-98 {np.mean([x[1] for x in g]):.2f}')
        for x in g[:8]: print('    ',x[1:])
