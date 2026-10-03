import json, sys, statistics
from pathlib import Path
for d in sys.argv[1:]:
    rows = [json.loads(l) for l in open(Path(d) / "log.jsonl")]
    last = rows[-1]
    ms = [r["ms_step"] for r in rows[2:] if "ms_step" in r]  # skip step 0 and the compile window
    if not ms and len(rows) > 2:  # pre-CUDA code: elapsed incl. eval
        ms = [1000 * (b["elapsed_s"] - a["elapsed_s"]) / (b["step"] - a["step"]) for a, b in zip(rows[1:], rows[2:])]
    h = last["heldout"]
    print(f"{Path(d).name:<12s} step {last['step']} ms/step {statistics.median(ms):6.2f} loss {last['train'].get('loss', 0):.4f} "
          f"heldout r2 2s/5s/10s/20s {h['r2_2s']:.4f}/{h['r2_5s']:.4f}/{h['r2_10s']:.4f}/{h['r2_20s']:.4f} "
          f"final {h['r2_final']:.4f} brier {h['brier']:.4f} select {last['select_score']:.4f}")
