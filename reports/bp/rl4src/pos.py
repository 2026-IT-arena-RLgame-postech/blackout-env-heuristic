import glob,numpy as np,sys
d='reports/bp/rl4/traj/'
def hist(prefix,t0,t1,who,cls_):
    H=np.zeros((25,25)); 
    for f in sorted(glob.glob(d+prefix+'__*.npz')):
        z=np.load(f)
        for j,sd in enumerate(z['sides']):
            tm=sd if who=='A' else 1-sd
            for t in range(t0,min(t1,(~np.isnan(z['pos'][j,:,0,0])).sum())):
                for u in range(5*tm,5*tm+5):
                    if z['cls'][j,t,u]==cls_:
                        x,y=z['pos'][j,t,u]
                        if tm==1: x,y=y,x   # into team-0 frame
                        H[int(np.clip(y,0,24)),int(np.clip(x,0,24))]+=1
    return H
for pre,who,lab in (('bp','A','RL4'),('bp','P','B5 vs RL4'),('B5','A','B5 mirror')):
    for cl,nm in ((1,'hunter'),(0,'collector')):
        H=hist(pre,110,800,who,cl); H/=H.sum()
        print('==',lab,nm,'110-800 (rows y desc, cols x), % per cell *10')
        for y in range(24,-1,-1): print(f"{y:2d} "+''.join(f"{int(round(H[y,x]*1000)):3d}" if H[y,x]>0.0005 else '  .' for x in range(25)))
