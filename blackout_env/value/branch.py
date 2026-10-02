"""S1.6: bootstrapped value targets on simulator branch data (fitted value iteration).

Data (examples/value_branch_data.py): each training row is a state ``s`` -- a 2 s branch end
(kind 1) or an on-trajectory state of a root match (kind 0) -- with the lead at +2 s and +5 s of a
fixed continuation (learner strategic_v23, opponent its own policy) and either the state ``s'``
5 s later or, when the match ended inside those 5 s, the exact final lead.  Every state is stored
from both teams' views (``own`` = the learner's, ``mir`` = the other team's), in the training
encoding of value/evaluate.py.

Targets (score/100 units, the learner's side; ``V_sym`` = the mirror-symmetric reading of the
previous iteration's net, evaluate.symmetrize):

  final(s)      = lead(s')  + rem_sym(s')        (= V_sym's final lead at s')     not ended
                = the final lead                                                   ended
  remaining(s)  = final(s) - lead(s)             (the net's final-lead head)
  delta_2s      = lead(+2 s) - lead(s)           exact (the final lead past the end)
  delta_5s      = lead(s') - lead(s)             exact
  delta_10s     = delta_5s + V_sym delta_5s(s')  bootstrapped; exact when ended
  delta_20s     = masked (no 15 s head to bootstrap from); exact when ended
  result probs  = V_sym's (loss, draw, win) at s'; one-hot of sign(final) when ended

The other team's view of ``s`` gets the antisymmetric target: every lead quantity negated, the
result probabilities reversed (``mirror_targets``).  Training on both views, with targets that
are antisymmetric by construction, removes the "the evaluated side is the strong learner" prior
the shard-trained S1 net carries.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blackout_env.value.targets import HORIZONS_S

LEAD_SCALE = 100.0  # data leads are in points; targets in score / 100
H2, H5, H10, H20 = (HORIZONS_S.index(h) for h in (2, 5, 10, 20))
VIEWS = ("own", "mir")


def bootstrap_targets(lead, lead2, lead5, ended, v_next: dict | None) -> dict[str, np.ndarray]:
    """Targets of N rows from the learner's side (module docstring).

    ``lead``, ``lead2``, ``lead5``: [N] points at s, +2 s, +5 s (for ended rows ``lead5`` is the
    final lead and ``lead2`` the lead at min(+2 s, end)); ``ended`` [N] bool.  ``v_next``: the
    symmetric prediction at s' (evaluate.symmetrize keys ``final`` [N] score/100, ``delta``
    [N, H], ``p`` [N, 3]); rows that ended ignore it (any values).  None = all rows ended."""
    lead = np.asarray(lead, np.float64) / LEAD_SCALE
    lead2 = np.asarray(lead2, np.float64) / LEAD_SCALE
    lead5 = np.asarray(lead5, np.float64) / LEAD_SCALE
    ended = np.asarray(ended, bool)
    n = len(lead)
    if v_next is None:
        if not ended.all():
            raise ValueError("v_next is required for rows that did not end")
        v_final, v_d5, v_p = np.zeros(n), np.zeros(n), np.zeros((n, 3))
    else:
        v_final = np.asarray(v_next["final"], np.float64)
        v_d5 = np.asarray(v_next["delta"], np.float64)[:, H5]
        v_p = np.asarray(v_next["p"], np.float64)
    final = np.where(ended, lead5, v_final)
    delta = np.zeros((n, len(HORIZONS_S)))
    mask = np.ones((n, len(HORIZONS_S)), bool)
    delta[:, H2] = lead2 - lead
    delta[:, H5] = lead5 - lead
    delta[:, H10] = np.where(ended, lead5 - lead, lead5 - lead + v_d5)
    delta[:, H20] = np.where(ended, lead5 - lead, 0.0)
    mask[:, H20] = ended
    onehot = np.zeros((n, 3))
    onehot[np.arange(n), np.sign(final).astype(int) + 1] = 1.0  # loss, draw, win
    p = np.where(ended[:, None], onehot, v_p)
    return {"lead": lead.astype(np.float32), "final": final.astype(np.float32),
            "remaining": (final - lead).astype(np.float32), "delta": delta.astype(np.float32),
            "delta_mask": mask, "p": p.astype(np.float32)}


def mirror_targets(t: dict) -> dict[str, np.ndarray]:
    """The same rows' targets from the other team's view: lead quantities negated, result
    probabilities reversed (loss <-> win)."""
    return {"lead": -t["lead"], "final": -t["final"], "remaining": -t["remaining"], "delta": -t["delta"],
            "delta_mask": t["delta_mask"].copy(), "p": t["p"][:, ::-1].copy()}


def load_raw(raw_dir: Path) -> dict:
    """All generated matches -> one flat dataset.  States are concatenated (``states[view]``:
    graphic / team_state / agent_states), rows index into them; ``seed`` per row for splits."""
    files = sorted(Path(raw_dir).glob("*.npz"))
    files = [f for f in files if not f.name.endswith(".tmp.npz")]
    rows = {k: [] for k in ("kind", "idx", "next_idx", "lead", "lead2", "lead5", "ended", "t_s", "root", "branch")}
    rows["seed"], rows["match"], rows["opponent"] = [], [], []
    states = {v: {"graphic": [], "team_state": [], "agent_states": []} for v in VIEWS}
    offset, opponents, metas = 0, [], []
    for m, f in enumerate(files):
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        metas.append(meta)
        n_states = len(z["s_g_own"])
        for k in ("kind", "idx", "next_idx", "lead", "lead2", "lead5", "ended", "t_s", "root", "branch"):
            v = z[f"r_{k}"]
            if k in ("idx", "next_idx"):
                v = np.where(v >= 0, v + offset, -1)
            rows[k].append(v)
        nr = len(z["r_kind"])
        opp = meta["task"]["opponent"]
        if opp not in opponents:
            opponents.append(opp)
        rows["seed"].append(np.full(nr, meta["task"]["seed"]))
        rows["match"].append(np.full(nr, m))
        rows["opponent"].append(np.full(nr, opponents.index(opp)))
        if n_states:
            for v in VIEWS:
                states[v]["graphic"].append(z[f"s_g_{v}"])
                states[v]["team_state"].append(z[f"s_ts_{v}"])
                states[v]["agent_states"].append(z[f"s_ag_{v}"])
        offset += n_states
    out_rows = {k: np.concatenate(v) for k, v in rows.items()}
    out_states = {v: {k: np.concatenate(a) for k, a in d.items()} for v, d in states.items()}
    return {"rows": out_rows, "states": out_states, "opponents": opponents, "metas": metas}


def holdout_mask(seed: np.ndarray, every: int = 20) -> np.ndarray:
    """Held-out rows: whole seeds (both sides of a seed -- V23 vs V23 sides mirror each other)."""
    return (np.asarray(seed) % every) == 0
