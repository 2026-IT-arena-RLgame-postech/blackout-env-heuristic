"""Merge R5 row sets pulled from several hosts (docs §59): merge_rows.py DEST SRC1 SRC2 ...
Every <set>_<base>/matches_*.jsonl is unioned by (tag, opponent, seed, side); the first copy wins."""
import json
import sys
from pathlib import Path

dest, srcs = Path(sys.argv[1]), [Path(s) for s in sys.argv[2:]]
files = sorted({f.relative_to(s) for s in srcs for f in s.glob("*_*/matches_*.jsonl")})
for rel in files:
    seen, out = set(), []
    for s in [dest] + srcs:
        f = s / rel
        if not f.exists():
            continue
        for x in f.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                k = (r["tag"], r["opponent"], r["seed"], r["side"])
                if k not in seen:
                    seen.add(k)
                    out.append(x)
    (dest / rel).parent.mkdir(parents=True, exist_ok=True)
    (dest / rel).write_text("\n".join(out) + "\n")
    print(f"{rel}: {len(out)}")
