import os,subprocess,time,glob
H=os.path.expanduser("~"); D=H+"/vitabc_eval"; V=H+"/blackout-env-vitab/checkpoints/distill"
st=[f"{a}_s{s}" for a in ("A","B1","B2","C1","C2") for s in (0,1)]; st4=[s for s in st if not s.startswith("A")]
opps={"p5a":"net:"+V+"/frozen/p5a_v22.pt","v19":"strategic_v19","p22":"net:"+V+"/frozen/p22_v31.pt","p24c":"net:"+V+"/frozen/p24c_v321.pt"}
jobs=[]
def J(cand,on,ospec,k):
    lo=960000+64*k; jobs.append((f"{cand}__{on}__{k}",f"{V}/vitab/{cand}/final.pt",ospec,lo,lo+64))
for on in ("p5a","v19"):
    for s in st+["C3_s0"]: J(s,on,opps[on],0)
for s in (0,1):
    for b,c in (("B2","C2"),("B1","C1")):
        for k in (0,1): J(f"{c}_s{s}",f"h2h-{b}_s{s}","net:"+f"{V}/vitab/{b}_s{s}/final.pt",k)
for on in ("p22","p24c"):
    for s in st4+["A_s0","A_s1","C3_s0"]: J(s,on,opps[on],0)
for k in (1,2,3):
    for on in ("p5a","v19"):
        for s in st: J(s,on,opps[on],k)
gpu={}
for l in open(D+"/jobs.txt"):
    p=l.split(); gpu[p[1]]=int(p[0])
def state(n): 
    if os.path.exists(f"{D}/{n}.json"): return "done"
    if os.path.exists(f"{D}/logs/{n}.log"): return "run"
    return "new"
g2=0
for j in jobs:
    n=j[0]
    while True:
        if os.path.exists(D+"/STOP"): raise SystemExit
        if state(n)!="new": break
        run=[x for x in gpu if state(x)=="run"]
        if len(run)<8:
            c=[sum(1 for x in run if gpu[x]==g) for g in (0,1)]
            g=0 if c[0]<=c[1] else 1; gpu[n]=g
            subprocess.Popen(["nohup",D+"/run1.sh",str(g),n,j[1],j[2],str(j[3]),str(j[4])],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
            time.sleep(1); break
        time.sleep(10)
