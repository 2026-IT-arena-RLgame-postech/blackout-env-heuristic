import glob,numpy as np,collections
d='reports/bp/rl4/traj/'
def run(prefix,label):
    ph=[(0,110),(110,400),(400,800),(800,1500),(1500,3000)]
    dm=collections.defaultdict(list); end=[];
    kl=collections.defaultdict(list); n=0
    for f in sorted(glob.glob(d+prefix+'__*.npz')):
        z=np.load(f)
        for j,(s,sd) in enumerate(zip(z['seeds'],z['sides'])):
            alive=~np.isnan(z['pos'][j,:,0,0]); T=alive.sum(); end.append(T); n+=1
            sc=z['score'][j][:T].astype(int); m=sc[:,sd]-sc[:,1-sd]
            m=np.concatenate([[0],m])
            fin=m[-1]
            for a,b in ph:
                a2=min(a,T); b2=min(b,T); dm[(a,b)].append(m[b2]-m[a2])
            k=z['kills'][z['kills'][:,0]==j]
            for _,t,kk,v,vc,kc in k:
                mine=((0 if v<5 else 1)==sd)
                for a,b in ph:
                    if a<=t<b: kl[(a,b,'lost' if mine else 'got',int(vc))].append(1)
    print(f'== {label} n={n} meanT={np.mean(end):.0f} final margin {np.mean([x for v in [dm[p] for p in ph] for x in []] or [sum(np.mean(dm[p]) for p in ph)]):+.1f}')
    print('  margin gain per phase:',{p:round(float(np.mean(dm[p])),1) for p in ph})
    for p in ph:
        print(f'   {p} kills/match: '+' '.join(f"{w}{c}:{len(kl[(p[0],p[1],w,c)])/n:.1f}" for w in ('got','lost') for c in (0,1,2)))
for p,l in [('B5_guard_trip0.8','trip v B5'),('B5','B5 v B5')]: run(p,l)
