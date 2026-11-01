import glob,numpy as np,collections,sys
d=sys.argv[1]; prefix=sys.argv[2]; t0=int(sys.argv[3]) if len(sys.argv)>3 else 220
tr=collections.Counter(); dur=collections.defaultdict(list); n=0
for f in sorted(glob.glob(d+prefix+'__*.npz')):
    z=np.load(f)
    for j,sd in enumerate(z['sides']):
        T=int((~np.isnan(z['pos'][j,:,0,0])).sum()); n+=1
        ho=(z['cls'][j,:T,5*sd:5*sd+5]==1).sum(1); hp=(z['cls'][j,:T,5*(1-sd):5*(1-sd)+5]==1).sum(1)
        s=ho*3+hp; t=t0
        # states, merge flickers < 10 ticks
        cur=s[t]; start=t
        for u in range(t+1,T):
            if s[u]!=cur:
                # look ahead: if the change reverts within 10 ticks ignore
                if (s[u:u+10]==cur).any() and False: continue
                tr[((cur//3,cur%3),(s[u]//3,s[u]%3))]+=1; dur[(cur//3,cur%3)].append((u-start)/50); cur=s[u]; start=u
print(prefix,'matches',n,'transitions per match from t>=',t0/50,'s')
src=collections.defaultdict(int)
for (a,b),v in tr.items(): src[a]+=v
for a in sorted(src):
    print(f" {a} n/match {src[a]/n:.1f} mean dur {np.mean(dur[a]):.1f}s ->",' '.join(f"{b}:{v/src[a]:.2f}" for (aa,b),v in sorted(tr.items(),key=lambda x:-x[1]) if aa==a and v/src[a]>=.04))
