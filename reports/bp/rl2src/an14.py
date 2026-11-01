# first post-opening transform tick per team after the opening trade; distribution + who first
import glob,numpy as np,collections,sys
d=sys.argv[1]; pref=sys.argv[2]
rows=[]
for f in sorted(glob.glob(d+pref+'__*.npz')):
    z=np.load(f)
    for j,(s,sd) in enumerate(zip(z['seeds'],z['sides'])):
        T=(~np.isnan(z['pos'][j,:,0,0])).sum()
        out=[]
        for tm in (sd,1-sd):
            sl=slice(5*tm,5*tm+5); c=z['cls'][j,:T,sl]; nh=(c==1).sum(1)
            # trade completes when nh first drops to 0 after >=100
            t0=next((t for t in range(95,T) if nh[t-1]>0 and nh[t]==0 or (t>100 and nh[t]==0)),None)
            first=None
            if t0 is not None:
                first=next((t for t in range(t0+1,T) if nh[t]>0),None)
            out.append(first)
        rows.append(out)
w=l=t=0; dts=[]
for a,b in rows:
    if a is None or b is None: continue
    if a<b: w+=1
    elif a>b: l+=1
    else: t+=1
    dts.append((a,b))
a=np.array(dts); print(pref,'n',len(rows),'both refilled',len(dts),'A first/tie/P first',w,t,l,'median first-refill tick A/P',np.median(a[:,0]),np.median(a[:,1]),'mean',a.mean(0))
