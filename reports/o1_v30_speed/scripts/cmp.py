import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
assert set(a) == set(b), (len(a), len(b))
bad = 0; n = 0
for k in sorted(a):
    ha, hb = a[k]["hashes"], b[k]["hashes"]; n += len(ha)
    if ha != hb:
        bad += 1
        i = next((i for i, (x, y) in enumerate(zip(ha, hb)) if x != y), min(len(ha), len(hb)))
        print("DIVERGE", k, "tick", i, "of", len(ha))
ta = sum(v["act_s"] for v in a.values()); tb = sum(v["act_s"] for v in b.values())
print(f"{len(a)} seqs {n} ticks, {bad} diverging; act {ta:.2f}s -> {tb:.2f}s ({ta / max(tb, 1e-9):.2f}x), "
      f"{ta / n * 1e6:.0f} -> {tb / n * 1e6:.0f} us/tick")
