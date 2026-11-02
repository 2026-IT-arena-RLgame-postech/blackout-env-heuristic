import glob,sys,numpy as np
d=sys.argv[1]; idx=int(sys.argv[2]); t0,t1,st=int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])
fs=sorted(glob.glob(d+'/*.npz')); k=0
for f in fs:
    z=np.load(f)
    for j,sd in enumerate(z['sides']):
        if k==idx:
            print('seed',z['seeds'][j],'RL side',sd)
            for t in range(t0,t1,st):
                def fr(team,u):
                    x,y=z['pos'][j,t,5*team+u]
                    if team==1: x,y=y,x
                    return f"{'CHC'[0] if 0 else 'cHC?'[z['cls'][j,t,5*team+u]]}({x:4.1f},{y:4.1f})" if not np.isnan(x) else '  --  '
                # frame: team1 mirrored to team0 frame
                print(t,'RL ',' '.join(fr(sd,u) for u in range(5)),'| OPP',' '.join(fr(1-sd,u) for u in range(5)))
            kk=z['kills'][z['kills'][:,0]==j]
            print([(int(t),int(a),int(v)) for _,t,a,v,vc,kc in kk if t<t1+400][:40])
        k+=1
