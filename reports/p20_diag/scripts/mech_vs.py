"""P20 (docs §47): per opponent, student vs base on the same (seed, side): second race w/t/l, Hunter seconds,
ticks, margin, and the margin delta split by (base 2nd-race outcome -> student outcome).
    python mech_vs.py --base p17 --ours 'p18bp2;p19' --opp 'V28;p17' matches.jsonl ..."""
import argparse, json, collections
import numpy as np
AL = {"V28": "strategic_v28:reserve_stage=0.5,trade_hold=advance",
      "V29": "strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve",
      "V30": "strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true",
      "V27": "strategic_v27:reserve_stage=1.5", "V25": "strategic_v25", "V23": "strategic_v23"}
ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--base", default="p17")
ap.add_argument("--ours", default="p18bp2;p19"); ap.add_argument("--opp", default="V28;p17")
a = ap.parse_args()
rows = [json.loads(x) for f in a.files for x in open(f) if x.strip()]
by = collections.defaultdict(dict)
for r in rows:
    by[(r["ours"], r["opponent"])][(r["seed"], r["side"])] = r
def oc(x):
    p, q = x["second_ours"], x["second_opp"]; p = 1e9 if p < 0 else p; q = 1e9 if q < 0 else q
    return "W" if p < q else ("T" if p == q else "L")
for o in [AL.get(s, s) for s in a.opp.split(";")]:
    for us in [a.base] + [AL.get(s, s) for s in a.ours.split(";")]:
        v = list(by.get((us, o), {}).values())
        if not v: continue
        f = lambda k: np.mean([x[k] for x in v])
        c = collections.Counter(oc(x) for x in v)
        print(f"{us[:24]:24s} vs {o[:24]:24s} n {len(v):3d} 2nd W/T/L {c['W']}/{c['T']}/{c['L']} | Hs ours {f('hunter_s'):5.1f} "
              f"theirs {f('opp_hunter_s'):5.1f} | ticks {f('ticks'):5.0f} | margin {f('margin'):+5.1f} | res100 {f('reserve100'):.1f}")
    base = by.get((a.base, o), {})
    for us in [AL.get(s, s) for s in a.ours.split(";")]:
        d = by.get((us, o), {}); ks = sorted(set(d) & set(base))
        if not ks: continue
        cat = collections.defaultdict(list)
        for k in ks: cat[oc(base[k]) + ">" + oc(d[k])].append(d[k]["margin"] - base[k]["margin"])
        tot = sum(sum(v) for v in cat.values()) / len(ks)
        print(f"  {us[:20]} - {a.base}: {tot:+.1f}/match; by 2nd race base>ours (n, mean d, share of total):",
              " ".join(f"{k}:{len(v)},{np.mean(v):+.1f},{sum(v)/len(ks):+.1f}" for k, v in sorted(cat.items())))
