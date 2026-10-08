import json, sys, statistics
for f in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f) if l.startswith("{")]
    if not rows:
        print(f, "no rows"); continue
    last = rows[-1]
    ms = [r["ms_step"] for r in rows[1:] if "ms_step" in r and "val_hit" not in r]
    if not ms and len(rows) > 2:  # ref script: cumulative seconds
        ms = [1000 * (b["s"] - a["s"]) / (b["step"] - a["step"]) for a, b in zip(rows[1:], rows[2:]) if "val_hit" not in b]
    val = {k[4:]: round(v, 4) for k, v in last.items() if k.startswith("val_")}
    print(f"{f.split('/')[-1]:<22s} step {last['step']} loss {last['loss']:.5f} median ms/step {statistics.median(ms) if ms else float('nan'):.2f} total {last['s']:.0f}s val {val}")
