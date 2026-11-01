import json,sys,numpy as np
for f in sys.argv[1:]:
    r=[json.loads(l) for l in open(f)]
    print(f,len(r),'keys' if 0 else '')
    for sd in (0,1):
        m=np.array([x['margin'] for x in r if x['side']==sd]); print(' side',sd,len(m),f"{m.mean():+.1f}±{m.std(ddof=1)/np.sqrt(len(m)):.1f} win {(m>0).mean():.2f}")
    m=np.array([x['margin'] for x in r]);print(f" all {m.mean():+.1f}±{m.std(ddof=1)/len(m)**.5:.1f}")
