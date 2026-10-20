import json,glob,os,sys,numpy as np
D=os.path.expanduser("~/vitabc_eval"); 
def load(name,chunks):
    out=[]
    for k in chunks:
        f=f"{D}/{name}__{k}.json"
        if not os.path.exists(f): return None
        d=json.load(open(f)); out.append(np.array(d["margin_side0"]+d["margin_side1"]))  # not interleaved; order consistent per file
    # re-order: concat per side so chunks align; keep (chunk,side) layout
    return np.concatenate(out)
def win(m): return np.where(m>0,1.0,np.where(m==0,0.5,0.0))
def se(x): return x.std(ddof=1)/np.sqrt(len(x))
arms=["A","B1","B2","C1","C2"]
def chunks_done(opp):
    ks=[]
    for k in range(4):
        if all(os.path.exists(f"{D}/{a}_s{s}__{opp}__{k}.json") for a in arms for s in (0,1)): ks.append(k)
        else: break
    return ks
res={}
print("## per-arm vs opponents (mean over 2 training seeds; SE over games of seed-averaged margin)")
for opp in ("p5a","v19","p22","p24c"):
    ks=chunks_done(opp)
    if not ks: continue
    n=len(ks)*64
    print(f"\n### {opp}: {n} seeds x 2 sides")
    print("| arm | win | margin +- SE | s0 margin | s1 margin | s0-s1 spread |"); print("|---|---|---|---|---|---|")
    M={}
    for a in arms:
        m=[load(f"{a}_s{s}__{opp}",ks) for s in (0,1)]; M[a]=m
        avg=(m[0]+m[1])/2; w=(win(m[0])+win(m[1]))/2
        print(f"| {a} | {w.mean():.3f} +- {se(w):.3f} | {avg.mean():+.1f} +- {se(avg):.1f} | {m[0].mean():+.1f} +- {se(m[0]):.1f} | {m[1].mean():+.1f} +- {se(m[1]):.1f} | {m[0].mean()-m[1].mean():+.1f} |")
    c3=load(f"C3_s0__{opp}",ks)
    if c3 is not None: print(f"| C3 (s0 only) | {win(c3).mean():.3f} +- {se(win(c3)):.3f} | {c3.mean():+.1f} +- {se(c3):.1f} | | | |")
    print("\npaired diffs (same seeds/sides, seed-averaged):"); print("| diff | dWin +- SE | dMargin +- SE |"); print("|---|---|---|")
    for x,y in (("B1","A"),("B2","A"),("C1","B1"),("C2","B2"),("C1","B2"),("C2","B1")):
        dm=(M[x][0]+M[x][1])/2-(M[y][0]+M[y][1])/2; dw=(win(M[x][0])+win(M[x][1]))/2-(win(M[y][0])+win(M[y][1]))/2
        print(f"| {x}-{y} | {dw.mean():+.3f} +- {se(dw):.3f} | {dm.mean():+.1f} +- {se(dm):.1f} |")
    # pooled B vs C
    B=np.mean([M[a][s] for a in ("B1","B2") for s in (0,1)],axis=0); C=np.mean([M[a][s] for a in ("C1","C2") for s in (0,1)],axis=0)
    d=C-B; print(f"| C(C1,C2)-B(B1,B2) | | {d.mean():+.1f} +- {se(d):.1f} |")
print("\n## head-to-head (row arm as 'ours' = C vs B opponent; margin positive = C better)")
for b,c in (("B2","C2"),("B1","C1")):
    for s in (0,1):
        ks=[k for k in (0,1) if os.path.exists(f"{D}/{c}_s{s}__h2h-{b}_s{s}__{k}.json")]
        if len(ks)==2:
            m=load(f"{c}_s{s}__h2h-{b}_s{s}",ks); w=win(m)
            print(f"{c}_s{s} vs {b}_s{s}: 128 seeds x2: C win {w.mean():.3f} +- {se(w):.3f}, margin {m.mean():+.1f} +- {se(m):.1f}")
