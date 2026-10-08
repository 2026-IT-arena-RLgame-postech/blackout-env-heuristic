---
name: feedback-metric-choice-matters
description: "When benchmarking a change meant to speed up a pipeline, measure the metric the user actually cares about end-to-end, not a proxy that trivially inflates"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 2a52de8d-09c8-4cf3-9dd4-e524bdbef845
  modified: 2026-09-13T08:46:08.778Z
---

Reported a Unity multi-arena batching change as "~4x throughput" based on "agent-decisions/s" (every unit's decision counted). Built the full pipeline on that premise, then when actually wiring up the real consumer (a heuristic-vs-heuristic data collector) and measuring transitions/s reaching the replay buffer — the metric that actually matters for data collection — the real improvement was only ~1.07x. The proxy metric inflated automatically with arena count (4 arenas = 4x the units = 4x the "decisions") regardless of whether the underlying bottleneck (network round-trip overhead) was actually amortized; it wasn't, because a different cost (Unity's own per-arena FixedUpdate compute) scaled up right alongside it and cancelled most of the gain. See [[project_unity_throughput_scaling]] for the full story.

**Why:** proxy/intermediate metrics (things easy to measure in a synthetic benchmark: agent-decisions/s, ticks/s, bytes/s) can look great while the metric the user is actually optimizing for (wall-clock time to collect N training transitions, in this case) barely moves. The gap only became visible once the *actual end-to-end consumer* was built and measured, not the synthetic benchmark harness.

**How to apply:** before reporting a speedup as a headline number, ask "is this the metric the user will actually feel?" — for a data-collection pipeline that's transitions/s or wall-clock-per-dataset, not decisions/s or ticks/s. When a synthetic micro-benchmark and the real end-to-end path could plausibly diverge (anything involving batching, amortization, or "N things per round-trip" claims), build/run the real consumer before declaring victory, not just the isolated benchmark. If a large claim (e.g. "4x") was already made from a proxy metric, and the real end-to-end path becomes measurable, re-verify against it and be upfront about a correction rather than letting the earlier headline number stand uncorrected.
