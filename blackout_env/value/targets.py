"""Value targets of one distill shard (one match, learner's canonical team frame).

Every shard row is one learner decision tick (25 per second: ``team_state[2]`` = time left
drops by 1/10500 per tick, and ``tick`` counts them).  ``team_state[0] - team_state[1]`` is the
learner's score lead / 100 at that tick -- the frame is already canonical (team_frame.py), so
team B's matches are the same task as team A's and need no mirroring.

Targets per row, all in score / 100 units like ``team_state``:

  delta   [T, H]  lead at tick + 25 * N (N in HORIZONS_S) minus the lead now
  final   [T]     the lead at the match end (meta learner_score - opponent_score)
  result  scalar  the learner's result: 1 win, 0 draw, -1 loss (meta ``result``)

When the future tick lies beyond the match end, the lead is the final lead: a match ends when the
result is decided (a decided score, the 420 s clock, or ``stop_when_exhausted`` = no battery left
anywhere after an absorption) and nothing can change afterwards.  The match end is
``max(last row tick, round(elapsed_s * 25))``; the final lead is appended there as a virtual row.

Inside the match the lead is read off the row at exactly that tick.  Shards written before
2026-09-30 dropped unlabelled ticks, so a tick can be missing: then the lead is known only when
the rows on both sides of the gap agree (it is piecewise constant, changing at deliveries,
steals and kills of carriers); otherwise ``delta_mask`` is False.  A shard whose ticks cannot be
reconstructed (``tick`` = -1) has every delta masked; its final/result targets still hold.
"""

from __future__ import annotations

import numpy as np

from blackout_env.train.distill.dataset import UNKNOWN_TICK, reconstruct_ticks

DECISIONS_PER_S = 25
MATCH_SECONDS = 420
HORIZONS_S = (2, 5, 10, 20)
RESULT_CLASSES = (-1, 0, 1)
LEAD_TOL = 1e-4  # leads are whole points / 100; float32 rows vs the float64 meta final  # class index = result + 1: loss, draw, win


def shard_ticks(shard) -> np.ndarray:
    """The ``tick`` column, reconstructed from time left for shards without one."""
    if "tick" in shard:
        return np.asarray(shard["tick"], dtype=np.int64)
    return reconstruct_ticks(np.asarray(shard["team_state"])).astype(np.int64)


def final_lead(meta: dict) -> float:
    return (float(meta["learner_score"]) - float(meta["opponent_score"])) / 100.0


def end_tick(ticks: np.ndarray, meta: dict) -> int:
    return int(max(int(ticks[-1]), round(float(meta.get("elapsed_s", 0.0)) * DECISIONS_PER_S)))


def lead_at(ticks: np.ndarray, lead: np.ndarray, query: np.ndarray, end: int, final: float
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lead at ``query`` ticks: (value, known, past_end).

    ``ticks`` strictly increasing, ``lead`` the per-row lead, ``end`` >= ticks[-1] the match end,
    where the lead becomes ``final`` (a virtual row); past it the lead stays ``final``."""
    t, v = (np.append(ticks, end), np.append(lead, final)) if end > ticks[-1] else (ticks, lead)
    past = query >= end
    j = np.clip(np.searchsorted(t, query, side="left"), 0, len(t) - 1)
    exact = t[j] == query
    prev = np.clip(j - 1, 0, len(t) - 1)
    bracketed = (j > 0) & (np.abs(v[prev] - v[j]) < LEAD_TOL)  # a gap with the same lead on both sides
    value = np.where(past, final, v[j])
    known = past | exact | bracketed
    return value.astype(np.float32), known, past


def match_targets(team_state: np.ndarray, ticks: np.ndarray, meta: dict,
                  horizons_s=HORIZONS_S) -> dict[str, np.ndarray]:
    """Per-row targets of one match (module docstring)."""
    team_state = np.asarray(team_state, dtype=np.float32)
    ticks = np.asarray(ticks, dtype=np.int64)
    n = len(team_state)
    lead = team_state[:, 0] - team_state[:, 1]
    final = final_lead(meta)
    delta = np.zeros((n, len(horizons_s)), np.float32)
    mask = np.zeros((n, len(horizons_s)), bool)
    past = np.zeros((n, len(horizons_s)), bool)
    valid_ticks = n > 0 and ticks[0] != UNKNOWN_TICK and (n == 1 or np.all(np.diff(ticks) > 0))
    if valid_ticks:
        end = end_tick(ticks, meta)
        for h, seconds in enumerate(horizons_s):
            value, known, beyond = lead_at(ticks, lead, ticks + seconds * DECISIONS_PER_S, end, final)
            delta[:, h] = value - lead
            mask[:, h] = known
            past[:, h] = beyond
    delta[~mask] = 0.0
    elapsed = (1.0 - team_state[:, 2].astype(np.float64)) * MATCH_SECONDS
    return {
        "lead": lead,
        "delta": delta,
        "delta_mask": mask,
        "past_end": past,
        "final": np.full(n, final, np.float32),
        "final_delta": (final - lead).astype(np.float32),
        "result": np.full(n, int(meta["result"]), np.int8),
        "elapsed_s": elapsed.astype(np.float32),
    }
