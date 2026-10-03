"""S4 Unity check: a deeper/wider search variant (examples/unity_search_eval.py --deep ...) paired
against the §21 rows of the same (opponent, seed, side) -- plain r5 and the S3 default search
(reports/unity_search/s8e25m3_{p11,v23}).  docs/policy_v2_design_ko.md §26.

    python examples/s4_unity_compare.py reports/s4_unity/t2bf --tag t2bf
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def se(x) -> float:
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def race(r) -> int:
    a, b = r["first_ours"], r["first_opp"]
    a = 10 ** 9 if a < 0 else a
    b = 10 ** 9 if b < 0 else b
    return int(np.sign(b - a))


def paired(new: dict, ref: dict) -> tuple[float, float, int]:
    keys = sorted(k for k in new if k in ref)
    by = defaultdict(list)
    for k in keys:
        by[k[1]].append(new[k]["margin"] - ref[k]["margin"])
    per = [np.mean(v) for v in by.values()]
    return float(np.mean(per)) if per else float("nan"), se(per), len(keys)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out")
    p.add_argument("--tag", required=True)
    p.add_argument("--ref-dir", default=str(REPO / "reports/unity_search"))
    a = p.parse_args()
    rows = load(Path(a.out) / "matches.jsonl")
    lines = []
    for opp in ("p11", "v23"):
        new = {(r["opponent"], r["seed"], r["side"]): r for r in rows if r["tag"] == a.tag and r["opponent"] == opp}
        if not new:
            continue
        ref_rows = load(Path(a.ref_dir) / f"s8e25m3_{opp}" / "matches.jsonl")
        plain = {(r["opponent"], r["seed"], r["side"]): r for r in ref_rows if r["tag"] == "plain"}
        s3 = {(r["opponent"], r["seed"], r["side"]): r for r in ref_rows if r["tag"] == "search"}
        d_plain, s_plain, n = paired(new, plain)
        d_s3, s_s3, _ = paired(new, s3)
        d_s3p, s_s3p, _ = paired({k: s3[k] for k in new if k in s3}, plain)
        rc = lambda d: tuple(sum(race(d[k]) == v for k in new if k in d) for v in (1, 0, -1))  # noqa: E731
        vals = list(new.values())
        lines.append(
            f"{a.tag} vs {opp}: n={n}  {a.tag} {np.mean([r['margin'] for r in vals]):+.1f}  "
            f"- r5 {d_plain:+.1f} ± {s_plain:.1f}  - S3 default {d_s3:+.1f} ± {s_s3:.1f}  "
            f"(S3 - r5 on these matches {d_s3p:+.1f} ± {s_s3p:.1f})  race w/t/l {rc(new)} vs S3 {rc(s3)} vs r5 {rc(plain)}  "
            f"win {np.mean([r['margin'] > 0 for r in vals]):.2f}  same as S3 {np.mean([new[k]['margin'] == s3[k]['margin'] for k in new if k in s3]):.2f}  "
            f"search {np.mean([r['search_ms_mean'] or 0 for r in vals]):.0f} ms (p95 {np.mean([r['search_ms_p95'] or 0 for r in vals]):.0f})  "
            f"search s/match {np.mean([r['search_s'] for r in vals]):.0f}  wall {np.mean([r['wall_s'] for r in vals]):.0f} s  "
            f"fallbacks {sum(r.get('fallbacks', 0) for r in vals)}  desyncs {sum(r['desyncs'] for r in vals)}")
        ch = np.sum([r["chosen"] for r in vals], 0)
        lines.append(f"   base share {ch[0] / ch.sum():.2f}")
    txt = "\n".join(lines)
    print(txt)
    (Path(a.out) / "compare.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
