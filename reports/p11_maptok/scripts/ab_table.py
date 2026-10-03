"""Markdown table of offline A/B arms (runs/<arm>/ab.json): held-out hit / angle error / within 2 deg
per time split, at every eval step.  python ab_table.py runs/ft_base runs/ft_f3 ..."""

import json
import sys
from pathlib import Path

SPLITS = ("all", "t0_10s", "t10_30s", "t30s_plus")


def cell(d):
    return f"{d['accuracy']:.4f} / {d['angle_err_deg/all']:.2f}° / {d['angle_within2/all']:.3f}"


def main():
    print("| arm | params | ms/step | step | " + " | ".join(f"{s} hit / err / 2°" for s in SPLITS) + " |")
    print("|---|---|---|---|" + "---|" * len(SPLITS))
    for arg in sys.argv[1:]:
        p = Path(arg)
        ab = json.loads((p / "ab.json").read_text())
        rows = [(e["step"], e) for e in ab.get("evals", [])] + [(ab["steps"], ab["heldout_splits"])]
        for step, e in rows:
            print(f"| {p.name} | {ab['params'] / 1e6:.2f}M | {ab['median_ms_step']:.0f} | {step} | "
                  + " | ".join(cell(e[s]) if s in e else "-" for s in SPLITS) + " |")


if __name__ == "__main__":
    main()
