import sys, glob, numpy as np
sys.path.insert(0,'examples'); sys.path.insert(0,'.')
from d1_why import load, HUNTER, F32
from pathlib import Path
out=sys.argv[1]
opps=sorted({Path(p).name.split('__')[1] for p in glob.glob(out+'/raw/*.npz')})
# training pool weights (P19 run_info): extra 0.5 over 19 descriptors; mixture 0.35; checkpoints 0.15 (offline QMIX, not simulated: no staged reserve assumed)
extra_n=19
print(f"{'opponent':40s} {'games':>5s} {'join@85-99 rows/g':>17s} {'hold@85-99 rows/g':>17s} {'games w/ join':>13s} {'h99=4':>6s}")
tot_join=0; tot_w=0
for o in opps:
    D=load(out,'v30',o); fl=D['flags'][0]; sh=D['shmv'][0].astype(F32); side=D['side'].astype(int); n=len(side)
    st=(fl[85:99]&2)>0; jn=(fl[85:99]&4)>0
    stand=np.linalg.norm(sh[85:99],axis=-1)<1e-3
    jrows=(st&jn&~stand).sum()/n; hrows=(st&stand).sum()/n
    gj=(st&jn&~stand).any((0,2)).mean()
    us=np.stack([np.arange(5)+5*s for s in side])
    h4=np.mean([(D['cls'][99,b,us[b]]==HUNTER).sum()==4 for b in range(n)])
    w=0.35 if o=='mixture' else 0.5/extra_n
    tot_join+=w*jrows; tot_w+=w
    print(f"{o:40s} {n:5d} {jrows:17.2f} {hrows:17.2f} {gj:13.2f} {h4:6.2f}")
# remaining extra entries not simulated: teacher:v21-teach perturb, d3b, d4 ckpts (3/19) and checkpoints 0.15 -> assumed no join rows
print(f"pool-weighted join rows (staged reserve, ticks 85-98, teacher moving in) per match: {tot_join:.3f}  (simulated weight {tot_w:.2f} of 1)")
print(f"unit-rows per training match ~ 35 s x 25 x 5 = {35*25*5}; join-row share ~ {tot_join/(35*25*5):.1e}")
