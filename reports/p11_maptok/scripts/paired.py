import json,collections,numpy as np,sys
D=sys.argv[1]
def name(c):
    if '/' not in c: return c
    p=c.split('/'); return p[-2] if p[-1]=='model.pt' else p[-1].replace('.pt','')
by=collections.defaultdict(dict)
for f in ('sim','sim_rep','sim_rep_b'):
    for l in open(f'{D}/{f}/games.jsonl'):
        d=json.loads(l); by[(name(d['candidate']),d['opponent'])][(d['seed'],d['side'])]=d
def pd(a,b,o):
    A,B=by[(a,o)],by[(b,o)]
    x=np.array([(A[(s,0)]['lead']+A[(s,1)]['lead'])/2-(B[(s,0)]['lead']+B[(s,1)]['lead'])/2 for s in range(1100000,1100500) if (s,0) in A and (s,1) in A and (s,0) in B and (s,1) in B])
    return f"{x.mean():+.2f}±{x.std(ddof=1)/np.sqrt(len(x)):.2f} (n={len(x)})"
for c in ('p11','ft_base','ft_f3','ft_base_s1','ft_f3_s1'):
    g=by[(c,'v23')]; r=collections.Counter(d['race'] for d in g.values())
    print(c,'v23 margin %+.1f'%np.mean([d['lead'] for d in g.values()]),'race W/T/L %d/%d/%d'%(r[1],r[0],r[-1]),'vs p11',pd(c,'p11','v23') if c!='p11' else '')
print('f3_s1 - base_s1', pd('ft_f3_s1','ft_base_s1','v23'))
print('f3 - base (seed0)', pd('ft_f3','ft_base','v23'))
a=[]
for s in range(1100000,1100500):
    f=lambda c:(by[(c,'v23')][(s,0)]['lead']+by[(c,'v23')][(s,1)]['lead'])/2
    a.append((f('ft_f3')+f('ft_f3_s1'))/2-(f('ft_base')+f('ft_base_s1'))/2)
a=np.array(a); print('2-seed mean f3-base v23 %+.2f±%.2f n=%d'%(a.mean(),a.std(ddof=1)/np.sqrt(len(a)),len(a)))
for o in ('v19','p5a'):
    print(o,'f3-base',pd('ft_f3','ft_base',o),'f3-p11',pd('ft_f3','p11',o),'base-p11',pd('ft_base','p11',o))
