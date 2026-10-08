---
name: feedback-use-gui-for-eval-runs
description: GUI-visible test runs are only worth it at low time_scale and only when the user explicitly asks; otherwise run headless/fast
metadata: 
  node_type: memory
  type: feedback
  originSessionId: f19a240b-0801-45c4-b34f-cbed9563e6bf
  modified: 2026-09-12T13:32:08.447Z
---

Default to headless (`no_graphics=True`) for BlackOutEnv test/evaluation runs, at whatever `time_scale` throughput needs (e.g. 200x for a diagnostic batch of many episodes). Only launch the Unity GUI (`no_graphics=False`) when the user explicitly asks to watch, and in that case also drop `time_scale` to something actually watchable (~3x) — a GUI window at 200x is pointless, frames fly by too fast to see anything.

**Why:** User first asked for GUI runs so they could "get insights" by watching (2026-09-12), but on seeing a 200x GUI run corrected that this combination is useless ("200배속이라 gui로 띄워도 의미가 없는것 같은데, 명시적으로 요청한 경우만 3배속 정도로 gui 띄우는게 적당해보임" — GUI only makes sense when explicitly requested, and then only around 3x speed).

**How to apply:** For any evaluate/benchmark script run (heuristic-vs-heuristic matches, reward alignment checks, etc.): default headless + high time_scale for throughput. Only switch to GUI + ~3x when the user's message explicitly asks to watch/see the match. Don't proactively offer GUI+high-speed as a compromise — it doesn't satisfy the actual need (watchability).
