import glob,sys,numpy as np
from collections import Counter
d=sys.argv[1]; t0,t1=int(sys.argv[2]),int(sys.argv[3])
C=Counter(); Cn=Counter(); n=0
def zone(x,y):
    if x<5 and y>=19: return 'ownbase'
    if np.hypot(x-11.5,y-11.5)<4: return 'sanctuary(<4)'
    if x<6 and 13<=y<19: return 'ownS-exit(x<6,y13-19)'
    if y>=17.5 and x>=5: return 'own-E-corr(y>17.5)'
    if x>=17 and y<8: return 'enemy-base-area'
    return 'other'
for f in sorted(glob.glob(d+'/*.npz')):
    z=np.load(f)
    for j,sd in enumerate(z['sides']):
        n+=1; pos=z['pos'][j]
        for _,t,a,v,vc,kc in z['kills'][z['kills'][:,0]==j]:
            if not(t0<=t<t1) or vc==1: continue
            vt=v//5; x,y=pos[t,v]
            if vt==1: x,y=y,x
            # normalise so that frame = victim's own team frame (own base at x<5,y>=19)
            C[('RL' if vt==sd else 'OPP', zone(x,y))]+=1
for k,v in sorted(C.items()): print(k,round(v/n,2))
