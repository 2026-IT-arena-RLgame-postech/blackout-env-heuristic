import sys,numpy as np
z=np.load(sys.argv[1]); j=int(sys.argv[2]); t0,t1,st=int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])
sd=int(z['sides'][j]); print('seed',z['seeds'][j],'side(RL)',sd)
for t in range(t0,t1,st):
    row=[]
    for tm,nm in ((sd,'RL'),(1-sd,'B5')):
        sl=slice(5*tm,5*tm+5); c=z['cls'][j,t,sl]; p=z['pos'][j,t,sl]; h=z['held'][j,t,sl]
        ds=np.hypot(p[:,0]-12,p[:,1]-12)
        s=' '.join(f"{'CHK'[0] if False else 'c' if cc==0 else 'H' if cc==1 else 'K'}{dd:4.1f}{'*' if hh>=0 else ' '}" for cc,dd,hh in zip(c,ds,h))
        row.append(f"{nm} [{s}] sc{z['score'][j,t,tm]:3d}")
    print(t,'  |  '.join(row))
