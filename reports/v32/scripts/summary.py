"""V32 (docs §58): compact table of a v27_eval matches.jsonl -- per opponent, each candidate's margin, paired
delta vs --base (seed-mean of both sides, SE over seeds), win rate, Hunter-seconds, Hunters at decision 99.
    python reports/v32/scripts/summary.py --out reports/v32/s1 --base H4
Names: aliases of names.sh (longest matching prefix, remainder shown as +k=v)."""
import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def aliases():
    out = subprocess.run(["bash", "-c", f"R=. ; . {HERE}/names.sh; for k in \"${{!N[@]}}\"; do echo \"$k=${{N[$k]}}\"; done"],
                         capture_output=True, text=True).stdout
    al = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    return sorted(al.items(), key=lambda kv: -len(kv[1]))


def short(d, al):
    for k, v in al:
        if d == v:
            return k
        if d.startswith(v + ","):
            rest = d[len(v) + 1:]
            rest = re.sub(r"exit_guards=1", "g", rest)
            rest = re.sub(r"guard_min_hunters=", "gm", rest)
            rest = re.sub(r"sweep=true", "sw", rest)
            return f"{k}+{rest}"
    return d if len(d) < 40 else d[:37] + "..."


def se(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--base", default="H4")
    a = p.parse_args()
    al = aliases()
    rows = [json.loads(x) for x in (Path(a.out) / "matches.jsonl").read_text().splitlines() if x.strip()]
    by = defaultdict(dict)
    for r in rows:
        by[(short(r["ours"], al), short(r["opponent"], al))][(r["seed"], r["side"])] = r
    opps = sorted({o for _, o in by})
    ours = sorted({c for c, _ in by}, key=lambda c: (c != a.base, c))
    lines = []
    for o in opps:
        lines.append(f"== vs {o}")
        base = by.get((a.base, o), {})
        for c in ours:
            d = by.get((c, o))
            if not d:
                continue
            v = list(d.values())
            m = [r["margin"] for r in v]
            h = np.array([r.get("h99_ours", -1) for r in v])
            ho = np.array([r.get("h99_opp", -1) for r in v])
            line = (f"  {c:34s} n={len(v):4d} {np.mean(m):+6.1f}±{se(m):4.1f} win {np.mean([x > 0 for x in m]):.2f} "
                    f"H-s {np.mean([r['hunter_s'] for r in v]):4.0f}/{np.mean([r['opp_hunter_s'] for r in v]):4.0f} "
                    f"H@99 {h.mean():.2f}/{ho.mean():.2f} (4: {np.mean(h == 4):.2f})")
            keys = [k for k in d if k in base]
            if c != a.base and keys:
                per = defaultdict(list)
                for k in keys:
                    per[k[0]].append(d[k]["margin"] - base[k]["margin"])
                ps = [np.mean(x) for x in per.values()]
                line += f"  Δ{a.base} {np.mean(ps):+6.1f}±{se(ps):4.1f} ({len(ps)}s)"
            lines.append(line)
    # mean over opponents of the paired delta (per seed, averaged across opponents)
    lines.append(f"== mean Δ{a.base} over opponents (per-seed mean over opponents, SE over seeds)")
    for c in ours:
        if c == a.base:
            continue
        per = defaultdict(list)
        for o in opps:
            d, base = by.get((c, o), {}), by.get((a.base, o), {})
            for k in d:
                if k in base:
                    per[k[0]].append(d[k]["margin"] - base[k]["margin"])
        if per:
            ps = [np.mean(x) for x in per.values()]
            lines.append(f"  {c:34s} {np.mean(ps):+6.1f}±{se(ps):4.1f} ({len(ps)}s)")
    txt = "\n".join(lines)
    print(txt)
    (Path(a.out) / "summary.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
