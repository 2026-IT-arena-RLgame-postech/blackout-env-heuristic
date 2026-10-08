---
name: feedback-plot-style
description: Heatmaps / Elo charts must reuse the earlier tournament look (RdYlGn heatmap with %, n, SE per cell + mean column; Elo dot chart with 95% intervals; LOO diagnostics)
metadata:
  type: feedback
---

User (2026-10-08): render win-rate heatmaps and similar images in the same format/colours as before.

**Why:** consistency with earlier reports (reports/elo_augment_20260918, heuristic_tournament_*).

**How to apply:** reuse `_heatmap` / `_elo_plot` in `examples/elo24.py` (copied from `examples/tournament_adaptive.py` style) and `examples/analyze_elo_diagnostics.py`; don't invent new colour maps.
