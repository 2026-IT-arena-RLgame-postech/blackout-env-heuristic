"""B1: old (BatchRollout) vs new (PipelinedRollout) blind_search_eval rows -- per-game identity and throughput.

    python reports/b1_batched_search/compare.py --old DIR --new DIR [--ref DIR] [--logs DIR] --out reports/b1_batched_search/compare.txt

``--ref``: other old-mode rows of the same seeds (e.g. R3's reports/r3_search/sim_p20) checked too.
``--logs``: <mode>_<opponent>.log files with the ``/usr/bin/time`` line ("TIME wall .. user .. sys ..").
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

KEYS = ("margin", "ours", "opp", "ticks", "first_ours", "first_opp", "searches", "chosen", "gain_sum", "lead_at")


def rows(d: Path, tag: str) -> dict:
    out = {}
    for f in sorted(d.glob("matches_*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["tag"] == tag:
                    out[(r["opponent"], r["seed"], r["side"])] = r
    return out


def chunks(rs) -> list[dict]:
    seen = {}
    for r in rs:
        seen[(r["chunk_wall_s"], r["search_s"])] = r
    return list(seen.values())


def cpu_time(logs: Path | None, mode: str, opp: str):
    if logs is None:
        return None
    f = logs / f"{mode}_{opp}.log"
    if not f.exists():
        return None
    m = re.search(r"TIME wall ([\d.]+) user ([\d.]+) sys ([\d.]+)", f.read_text())
    return None if not m else tuple(float(x) for x in m.groups())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--ref", default=None)
    p.add_argument("--logs", default=None)
    p.add_argument("--tag", default="o4f_v28r")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    old, new = rows(Path(a.old), a.tag), rows(Path(a.new), a.tag)
    ref = rows(Path(a.ref), a.tag) if a.ref else {}
    logs = Path(a.logs) if a.logs else None
    lines = [f"B1 old vs new ({a.tag}): per-game identity and throughput", ""]
    summary = {}
    for opp in sorted({k[0] for k in new}):
        kn = sorted(k for k in new if k[0] == opp)
        common = [k for k in kn if k in old]
        diff = {key: sum(old[k][key] != new[k][key] for k in common) for key in KEYS}
        rcommon = [k for k in kn if k in ref]
        rdiff = {key: sum(ref[k][key] != new[k][key] for k in rcommon) for key in KEYS}
        mo = np.array([old[k]["margin"] for k in common], float)
        mn = np.array([new[k]["margin"] for k in common], float)
        lines.append(f"[{opp}] games new {len(kn)}, common with old {len(common)}, with ref {len(rcommon)}")
        lines.append(f"  differing fields vs old: {diff}")
        if rcommon:
            lines.append(f"  differing fields vs ref: {rdiff}")
        if len(common):
            lines.append(f"  margin mean old {mo.mean():+.2f} new {mn.mean():+.2f}  identical games {int((mo == mn).sum())}/{len(common)}")
        st = {}
        for mode, rs in (("old", [old[k] for k in common]), ("new", [new[k] for k in kn])):
            if not rs:
                continue
            cs = chunks(rs)
            wall = sum(c["chunk_wall_s"] for c in cs)
            search = sum(c["search_s"] for c in cs)
            n = sum(c["chunk_n"] for c in cs)
            roots = sum(c["chunk_searches"] for c in cs)
            ct = cpu_time(logs, mode, opp)
            st[mode] = {"games": n, "wall_s": wall, "search_s": search, "roots": roots,
                        "s_per_game": wall / n, "search_s_per_game": search / n, "games_per_h": 3600 * n / wall,
                        "s_per_root": search / max(1, roots)}
            msg = (f"  {mode}: {n} games, wall {wall:.0f} s ({wall / n:.1f} s/game, search {search / n:.1f} s/game, "
                   f"{search / max(1, roots):.3f} s/root) = {3600 * n / wall:.0f} games/h/process")
            if ct:
                cpu = ct[1] + ct[2]
                st[mode] |= {"proc_wall_s": ct[0], "cpu_s": cpu, "games_per_cpu_h": 3600 * n / cpu}
                msg += f"; process CPU {cpu:.0f} s ({cpu / ct[0]:.2f} cores) = {3600 * n / cpu:.0f} games per CPU-core-hour"
            lines.append(msg)
        if "old" in st and "new" in st:
            lines.append(f"  speed-up: wall {st['old']['s_per_game'] / st['new']['s_per_game']:.2f}x, "
                         f"search {st['old']['search_s_per_game'] / st['new']['search_s_per_game']:.2f}x"
                         + (f", per CPU core {st['new']['games_per_cpu_h'] / st['old']['games_per_cpu_h']:.2f}x"
                            if "games_per_cpu_h" in st["old"] and "games_per_cpu_h" in st["new"] else ""))
        summary[opp] = {"diff_vs_old": diff, "diff_vs_ref": rdiff, "common": len(common), "ref_common": len(rcommon),
                        "timing": st}
        lines.append("")
    txt = "\n".join(lines)
    print(txt)
    Path(a.out).write_text(txt + "\n")
    Path(a.out).with_suffix(".json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
