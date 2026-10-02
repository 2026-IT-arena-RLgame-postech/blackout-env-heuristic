"""S1.5: can the S1 value net rank short-horizon branches from the same root state?
(docs/policy_v2_design_ko.md "S1.5 분기 순위 검증").  Everything runs in the S2 simulator.

  1. ``matches``  -- play root matches: strategic_v23 (the learner) vs a mix of opponents, seeds
     950000+, both sides; record each match's length.  Root ticks are drawn per time bucket
     (0-10, 10-30, 30-60, 60-150 s), balanced over buckets and opponents.
  2. ``branches`` -- replay each selected match to its root ticks (deterministic) and from each
     root let the learner play each of BRANCHES for W = 2 s (WINDOWS; (25, 50) adds the 1 s
     prefix), the opponent its own policy (history-dependent heuristics deep-copied at the root;
     the learner's history-dependent branch policies are shadowed through the root match on the
     learner's observations so they too carry history).  After the window both teams continue
     (learner: fresh strategic_v23, opponent: a copy of its instance) for R reseeded rollouts
     (common random numbers: rollout r uses the same reseed for every branch) to +H = 20 s and
     to the match end (cap: CAP_S = 60 s after the window).  V (all heads; plain, last-k-tick averages,
     mirror-symmetric), the hand-feature MLP and the lead are read at the window end, and V again
     after a separate 2 s / 5 s continuation rollout ("V after rollout").
  3. ``report``   -- within-root ranking metrics per bucket x W x predictor: pairwise concordance
     by ground-truth gap, Spearman per root, top-1 regret vs oracle / random / no search.

    PYTHONPATH=. .venv/bin/python examples/value_branch_eval.py all --workers 12 \
        --out reports/value_s1/branch
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DPS = 25  # decisions per second
BUCKETS = ((0, 10), (10, 30), (30, 60), (60, 150))
BUCKET_NAMES = ("0-10s", "10-30s", "30-60s", "60s+")
WINDOWS = (50,)  # decisions (2 s); (25, 50) also scores the 1 s prefix of the same window (2x cost)
H_S = 20
CAP_S = 60  # continuation cap past the window end ("final" = final lead, or the lead at the cap)
LEAF_ROLL = (50, 125)  # V after a separate continuation rollout of 2 s / 5 s
AVG_K = (1, 5, 10)
P10 = "ckpt:checkpoints/distill/frozen/p10_v22.pt"
OPPONENTS = ("strategic_v23", "strategic_v22", "strategic_v21_anyangle", "strategic_v17", "strategic_v19")
STATELESS = {"strategic_v23", "strategic_v22", "strategic_v21_anyangle", P10}  # verified: fresh == persistent
HISTORY = ("strategic_v17", "strategic_v19")  # history-dependent: shadowed / deep-copied
BRANCHES = ("v23", "v22", "v21aa", "v17", "v19", "v23_eps25", "v23_stand1", "v23_sticky8")
VALUE_CKPT = "checkpoints/value/frozen/s1b_value.pt"
BASELINES = "reports/value_s1/baseline_models.pt"

# ------------------------------------------------------------------------------ policies


class StandOne:
    """A policy with one own unit standing still (unit index fixed per root)."""

    def __init__(self, policy, unit: str):
        self.policy, self.unit = policy, unit

    def act(self, obs):
        acts = self.policy.act(obs)
        if self.unit in acts:
            acts[self.unit] = np.zeros(2, np.float32)
        return acts


def _make(desc: str, seed: int = 0):
    from blackout_env.train.distill.opponents import make_opponent

    return make_opponent(desc, seed)


def branch_policy(name: str, shadows: dict, rng_seed: int, learner_team: int):
    from blackout_env.train.distill.perturb import PerturbedPolicy

    if name == "v23":
        return _make("strategic_v23")
    if name == "v22":
        return _make("strategic_v22")
    if name == "v21aa":
        return _make("strategic_v21_anyangle")
    if name == "v17":
        return copy.deepcopy(shadows["strategic_v17"])
    if name == "v19":
        return copy.deepcopy(shadows["strategic_v19"])
    if name == "v23_eps25":
        return PerturbedPolicy(_make("strategic_v23"), seed=rng_seed, eps=0.25)
    if name == "v23_sticky8":
        return PerturbedPolicy(_make("strategic_v23"), seed=rng_seed, sticky=8)
    if name == "v23_stand1":
        unit = 5 * learner_team + int(np.random.default_rng(rng_seed).integers(5))
        return StandOne(_make("strategic_v23"), f"unit_{unit}")
    raise ValueError(name)


def clone_opponent(desc: str, policy):
    return policy if desc == P10 else (_make(desc) if desc in STATELESS else copy.deepcopy(policy))


def team_names(team: int) -> list[str]:
    return [f"unit_{i}" for i in range(5 * team, 5 * team + 5)]


def joint_moves(obs, learner, opponent, learner_team: int) -> np.ndarray:
    acts = learner.act({n: obs[n] for n in team_names(learner_team)})
    acts.update(opponent.act({n: obs[n] for n in team_names(1 - learner_team)}))
    mv = np.zeros((10, 2), np.float32)
    for n, a in acts.items():
        mv[int(n.split("_")[1])] = np.clip(np.asarray(a, np.float32), -1, 1)
    return mv


def lead_of(sim, team: int) -> int:
    s = sim.scores
    return s[team] - s[1 - team]


def result_of(sim, team: int) -> float:
    lead = lead_of(sim, team)
    return 1.0 if lead > 0 else (0.0 if lead < 0 else 0.5)


# ------------------------------------------------------------------------------ stage 1: matches


def match_tasks(n_seeds: int, seed0: int) -> list[dict]:
    opps = list(OPPONENTS) * 2 + [P10]  # P10 student ~1/11 of matches (slow)
    tasks = []
    for j in range(n_seeds):
        for side in (0, 1):
            tasks.append({"id": f"{seed0 + j}_{'AB'[side]}", "seed": seed0 + j, "learner_team": side,
                          "opponent": opps[j % len(opps)]})
    return tasks


def play_root_match(task: dict, stop_tick: int | None = None, on_tick=None, shadows: bool = False) -> dict:
    """Plays V23 (learner) vs the opponent from the seed; ``on_tick(tick, sim, opp, shadows)`` is
    called before each decision.  Stops at the end, the cap (150 s) or ``stop_tick``."""
    from blackout_env.sim import SimState

    team = task["learner_team"]
    sim = SimState.from_seed(task["seed"])
    learner, opp = _make("strategic_v23"), _make(task["opponent"], task["seed"])
    shadow = {h: _make(h) for h in HISTORY} if shadows else {}
    last = BUCKETS[-1][1] * DPS if stop_tick is None else stop_tick
    tick = 0
    while not sim.over and tick < last:
        if on_tick is not None:
            on_tick(tick, sim, opp, shadow)
        obs = sim.obs()
        own = {n: obs[n] for n in team_names(team)}
        for h in shadow.values():
            h.act(own)
        sim.step(joint_moves(obs, learner, opp, team))
        tick += 1
    return {"ticks": tick, "over": sim.over, "lead": lead_of(sim, team), "scores": sim.scores,
            "pos_hash": int(zlib.crc32(sim.uf.tobytes()))}


def _match_worker(task: dict) -> dict:
    _init_threads()
    res = play_root_match(task)
    return {**task, **res}


# ------------------------------------------------------------------------------ stage 2: branches

_EV = None  # worker: (ValueEvaluator, baseline models)


def _init_threads():
    import torch

    torch.set_num_threads(1)


def _init_branch_worker(value_ckpt: str, baselines: str):
    global _EV
    _init_threads()
    from blackout_env.value.baselines import load_models
    from blackout_env.value.evaluate import ValueEvaluator

    _EV = (ValueEvaluator(str(REPO / value_ckpt), device="cpu", batch=128), load_models(REPO / baselines))


def _first_hunter(sim, team: int):
    from blackout_env.sim.mapdata import HUNTER

    return bool((sim.ui[5 * team: 5 * team + 5, 0] == HUNTER).any())


class RaceClock:
    def __init__(self, sim, team, tick):
        self.team, self.first = team, [tick if _first_hunter(sim, team) else None,
                                       tick if _first_hunter(sim, 1 - team) else None]

    def update(self, sim, tick):
        for i, t in enumerate((self.team, 1 - self.team)):
            if self.first[i] is None and _first_hunter(sim, t):
                self.first[i] = tick


def continuation(sim0, opp_desc, opp0, team: int, reseed: int, tick0: int, max_steps: int, race=None,
                 snap_at=()) -> dict:
    """Both teams continue (learner fresh V23, opponent a copy) from ``sim0``; returns lead at +H,
    final lead / result (or at the cap), and obs at the ``snap_at`` step counts."""
    sim = sim0.copy()
    sim.reseed(reseed)
    learner, opp = _make("strategic_v23"), clone_opponent(opp_desc, opp0)
    lead_h, snaps, step = None, {}, 0
    while not sim.over and step < max_steps:
        if step in snap_at and not sim.over:
            snaps[step] = sim.obs()
        if step == H_S * DPS:
            lead_h = lead_of(sim, team)
        sim.step(joint_moves(sim.obs(), learner, opp, team))
        step += 1
        if race is not None:
            race.update(sim, tick0 + step)
    if step in snap_at and step not in snaps and not sim.over:
        snaps[step] = sim.obs()
    final = lead_of(sim, team)
    return {"lead_h": final if lead_h is None else lead_h, "final": final, "win": result_of(sim, team),
            "capped": not sim.over, "steps": step, "snaps": snaps, "over": sim.over}


def run_root(sim_root, opp_desc, opp_root, shadows, team: int, tick: int, rng_seed: int, n_rollouts: int) -> dict:
    """All branches x windows from one root.  Returns ground truth and the raw views to evaluate."""
    need = {w: set(range(w - max(AVG_K) + 1, w + 1)) for w in WINDOWS}
    out = {"branches": []}
    views = []  # (key, obs, team) to evaluate with V
    for b, name in enumerate(BRANCHES):
        pol = branch_policy(name, shadows, rng_seed * 31 + b, team)
        opp = clone_opponent(opp_desc, opp_root)
        sim = sim_root.copy()
        race = RaceClock(sim, team, tick)
        ends, recorded = {}, {}
        step = 0
        while step < WINDOWS[-1] and not sim.over:
            sim.step(joint_moves(sim.obs(), pol, opp, team))
            step += 1
            race.update(sim, tick + step)
            if any(step in s for s in need.values()):
                recorded[step] = sim.obs()
            if step in WINDOWS:
                ends[step] = (sim.copy(), clone_opponent(opp_desc, opp), step)
        for w in WINDOWS:
            if w not in ends:  # the match ended inside the window: terminal from then on
                ends[w] = (sim.copy(), opp, step)
        rec = {"name": name, "windows": {}}
        for w in WINDOWS:
            sim_w, opp_w, reached = ends[w]
            row = {"terminal": bool(sim_w.over), "reached": reached, "lead_end": lead_of(sim_w, team),
                   "pos_hash": int(zlib.crc32(sim_w.uf.tobytes() + sim_w.ui.tobytes())),
                   "final_r": [], "lead_h_r": [], "win_r": [], "capped": 0}
            if sim_w.over:
                f, wn = lead_of(sim_w, team), result_of(sim_w, team)
                row["final_r"] = [f] * n_rollouts
                row["lead_h_r"] = [f] * n_rollouts
                row["win_r"] = [wn] * n_rollouts
            else:
                for r in range(n_rollouts):
                    c = continuation(sim_w, opp_desc, opp_w, team, 1_000_003 * (r + 1) + rng_seed, tick + reached,
                                     CAP_S * DPS, race=race if (r == 0 and w == WINDOWS[-1]) else None)
                    row["final_r"].append(c["final"])
                    row["lead_h_r"].append(c["lead_h"])
                    row["win_r"].append(c["win"])
                    row["capped"] += int(c["capped"])
                # V after a separate continuation rollout (its own reseed: not one of the R)
                leaf = continuation(sim_w, opp_desc, opp_w, team, 7_777_777 + rng_seed, tick + reached,
                                    max(LEAF_ROLL), snap_at=set(LEAF_ROLL))
                row["leaf_terminal"] = {}
                for k in LEAF_ROLL:
                    if k in leaf["snaps"]:
                        views.append(((b, w, f"roll{k}"), leaf["snaps"][k]))
                    else:  # ended before k: exact
                        row["leaf_terminal"][str(k)] = [leaf["final"], leaf["win"]]
                for k in need[w]:
                    if k in recorded:
                        views.append(((b, w, f"t{w - k}"), recorded[k]))
            rec["windows"][str(w)] = row
        rec["race"] = race.first
        out["branches"].append(rec)
    out["race_root"] = RaceClock(sim_root, team, tick).first
    return out, views


def evaluate_views(views, team: int) -> dict:
    """V (own + mirror readings), hand MLP and lead for every recorded view; key -> dict of scalars (points)."""
    from blackout_env.value.baselines import predict_set
    from blackout_env.value.evaluate import encode_batch, team_view
    from blackout_env.value.features import hand_features

    ev, models = _EV
    if not views:
        return {}
    own = encode_batch([team_view(o, team) for _, o in views])
    mir = encode_batch([team_view(o, 1 - team) for _, o in views])
    po = ev.evaluate_encoded(own["graphic"], own["team_state"], own["agent_states"])
    pm = ev.evaluate_encoded(mir["graphic"], mir["team_state"], mir["agent_states"])
    hm = predict_set("hand_mlp", models, hand_features(own["graphic"], own["team_state"], own["agent_states"]))
    out = {}
    for i, (key, _) in enumerate(views):
        out[key] = {
            "lead": float(po["lead"][i] * 100),
            "v_final": float(po["final"][i] * 100), "m_final": float(pm["final"][i] * 100),
            "v_d": [float(x * 100) for x in po["delta"][i]], "m_d": [float(x * 100) for x in pm["delta"][i]],
            "v_p": [float(x) for x in po["p"][i]], "m_p": [float(x) for x in pm["p"][i]],
            "h_final": float(hm["final"][i] * 100), "h_d20": float(hm["delta"][i][-1] * 100),
            "h_pwin": float(hm["p_win"][i]),
        }
    return out


def _branch_worker(job: dict) -> dict:
    task, root_ticks, n_rollouts = job["task"], job["root_ticks"], job["n_rollouts"]
    from blackout_env.sim import SimState  # noqa: F401

    team = task["learner_team"]
    snaps = {}

    def on_tick(tick, sim, opp, shadow):
        if tick in root_ticks:
            snaps[tick] = (sim.copy(), clone_opponent(task["opponent"], opp),
                           {k: copy.deepcopy(v) for k, v in shadow.items()})

    started = time.time()
    play_root_match(task, stop_tick=max(root_ticks) + 1, on_tick=on_tick, shadows=True)
    roots = []
    for tick in root_ticks:
        if tick not in snaps:
            continue
        sim_root, opp_root, shadows = snaps[tick]
        rng_seed = zlib.crc32(f"{task['id']}:{tick}".encode()) % 1_000_000
        gt, views = run_root(sim_root, task["opponent"], opp_root, shadows, team, tick, rng_seed, n_rollouts)
        root_obs = sim_root.obs()
        preds = evaluate_views(views + [(("root",), root_obs)], team)
        gt["preds"] = {"|".join(map(str, k)): v for k, v in preds.items()}
        gt.update({"match": task["id"], "seed": task["seed"], "learner_team": team, "opponent": task["opponent"],
                   "tick": tick, "t_s": tick / DPS, "lead_root": lead_of(sim_root, team)})
        roots.append(gt)
    return {"task": task, "roots": roots, "seconds": time.time() - started,
            "config": {"windows": WINDOWS, "cap_s": CAP_S, "h_s": H_S, "rollouts": n_rollouts, "branches": BRANCHES}}


# ------------------------------------------------------------------------------ root selection


def bucket_of(t_s: float) -> int:
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= t_s < hi:
            return i
    return -1


def select_roots(matches: list[dict], per_bucket: int, seed: int = 0) -> dict[str, list[int]]:
    """For each bucket, up to ``per_bucket`` roots, one per match, round-robin over opponents; the
    root tick is uniform over the part of the bucket the match reaches (leaving >= 1 window)."""
    rng = np.random.default_rng(seed)
    chosen: dict[str, list[int]] = {}
    for b, (lo, hi) in enumerate(BUCKETS):
        by_opp = {}
        for m in matches:
            top = min(hi * DPS, m["ticks"] - WINDOWS[0])
            if top > lo * DPS:
                by_opp.setdefault(m["opponent"], []).append((m, lo * DPS, top))
        for v in by_opp.values():
            rng.shuffle(v)
        picked, keys = 0, sorted(by_opp)
        while picked < per_bucket and any(by_opp.values()):
            for k in keys:
                if by_opp[k] and picked < per_bucket:
                    m, a, z = by_opp[k].pop()
                    chosen.setdefault(m["id"], []).append(int(rng.integers(a, z)))
                    picked += 1
    return chosen


# ------------------------------------------------------------------------------ report


def predictor_values(root: dict, w: int) -> dict:
    """predictor -> (final [B], lead_h [B], p_win [B]) in points / probability; terminal branch ends
    and leaf rollouts that ended are exact."""
    P = root["preds"]
    B = len(root["branches"])
    names = (["lead", "hand_mlp"] + [f"v_k{k}" for k in AVG_K] + [f"sym_k{k}" for k in AVG_K]
             + [f"v_roll{k}" for k in LEAF_ROLL] + [f"sym_roll{k}" for k in LEAF_ROLL]
             + [f"lead_roll{k}" for k in LEAF_ROLL])
    out = {n: np.full((3, B), np.nan) for n in names}
    for b, br in enumerate(root["branches"]):
        row = br["windows"][str(w)]
        if row["terminal"]:
            f = row["final_r"][0]
            for n in names:
                out[n][:, b] = (f, f, row["win_r"][0])
            continue

        def get(key):
            return P.get(f"{b}|{w}|{key}")

        last = get("t0")
        out["lead"][:, b] = (last["lead"], last["lead"], 0.5 + 0.5 * np.sign(last["lead"]))
        out["hand_mlp"][:, b] = (last["h_final"], last["lead"] + last["h_d20"], last["h_pwin"])
        for k in AVG_K:
            ps = [p for p in (get(f"t{j}") for j in range(k)) if p is not None]
            v = np.mean([(p["v_final"], p["lead"] + p["v_d"][-1], p["v_p"][2]) for p in ps], axis=0)
            s = np.mean([((p["v_final"] - p["m_final"]) / 2, p["lead"] + (p["v_d"][-1] - p["m_d"][-1]) / 2,
                          (p["v_p"][2] + p["m_p"][0]) / 2) for p in ps], axis=0)
            out[f"v_k{k}"][:, b], out[f"sym_k{k}"][:, b] = v, s
        for k in LEAF_ROLL:
            term = row.get("leaf_terminal", {}).get(str(k))
            p = get(f"roll{k}")
            if term is not None:
                out[f"v_roll{k}"][:, b] = out[f"sym_roll{k}"][:, b] = (term[0], np.nan, term[1])
                out[f"lead_roll{k}"][:, b] = (term[0], np.nan, term[1])
            else:
                out[f"lead_roll{k}"][:, b] = (p["lead"], np.nan, 0.5 + 0.5 * np.sign(p["lead"]))
                out[f"v_roll{k}"][:, b] = (p["v_final"], np.nan, p["v_p"][2])
                out[f"sym_roll{k}"][:, b] = ((p["v_final"] - p["m_final"]) / 2, np.nan, (p["v_p"][2] + p["m_p"][0]) / 2)
    return out


def _ties_argmax_mean(pred: np.ndarray, value: np.ndarray) -> float:
    """Expected ground truth of the predictor's pick, ties broken uniformly."""
    best = pred >= np.nanmax(pred) - 1e-9
    return float(value[best].mean())


def _rank(x):
    """Ranks with ties averaged."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):
        tie = x == v
        r[tie] = r[tie].mean()
    return r


def _spearman(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


TARGETS = ("final", "lead_h", "win")
GAP_THRESHOLDS = (0.0, 1.0, 3.0, 5.0)


def root_metrics(root: dict, w: int) -> dict:
    rows = [br["windows"][str(w)] for br in root["branches"]]
    gt_r = {"final": np.array([r["final_r"] for r in rows], float), "lead_h": np.array([r["lead_h_r"] for r in rows], float),
            "win": np.array([r["win_r"] for r in rows], float)}
    gt = {k: v.mean(1) for k, v in gt_r.items()}
    R = gt_r["final"].shape[1]
    preds = predictor_values(root, w)
    B = len(rows)
    iu = np.triu_indices(B, 1)
    out = {"gt": gt, "pairs": {}, "pick": {}, "spearman": {}}
    # noise of a pair gap under common random numbers
    for t in TARGETS:
        d = gt_r[t][iu[0]] - gt_r[t][iu[1]]
        out["pairs"][t] = {"gap": d.mean(1), "se": d.std(1, ddof=1) / np.sqrt(R) if R > 1 else np.zeros(len(d))}
    out["pred_pairs"] = {}
    for n, p in preds.items():
        for ti, t in enumerate(TARGETS):
            pv = p[ti]
            if np.isnan(pv).any():
                continue
            out["pred_pairs"][(n, t)] = np.sign(pv[iu[0]] - pv[iu[1]])
            out["spearman"][(n, t)] = _spearman(pv, gt[t])
        pick_by = p[0]  # the final-lead head decides; regret is in final-lead points
        out["pick"][n] = _ties_argmax_mean(pick_by, gt["final"])
        if not np.isnan(p[2]).any():
            out["pick"][n + "@pwin"] = _ties_argmax_mean(p[2], gt["final"])
            out["pick_win"] = out.get("pick_win", {})
            out["pick_win"][n] = _ties_argmax_mean(p[2], gt["win"])
    f = gt["final"]
    out["oracle"] = float(f.max())
    out["random"] = float(f.mean())
    out["no_search"] = float(f[BRANCHES.index("v23")])
    # cross-validated oracle: choose on half of the rollouts, value on the other half
    if R >= 2:
        h = R // 2
        a, b = gt_r["final"][:, :h].mean(1), gt_r["final"][:, h:].mean(1)
        out["oracle_cv"] = 0.5 * (_ties_argmax_mean(a, b) + _ties_argmax_mean(b, a))
    else:
        out["oracle_cv"] = out["oracle"]
    # V's calibration on the no-search branch (the continuation policy is V23 on both sides of the gap)
    b0 = BRANCHES.index("v23")
    out["calib"] = {n: float(gt["final"][b0] - preds[n][0][b0]) for n in ("hand_mlp", "v_k1", "sym_k1", "v_k10")}
    # opening race: who got the first Hunter (r = 0 continuation of the 2 s window), per branch
    if w == WINDOWS[-1]:
        margin = []
        for br in root["branches"]:
            own, opp = br["race"]
            own = 10**6 if own is None else own
            opp = 10**6 if opp is None else opp
            margin.append(np.sign(opp - own))  # +1 learner first, -1 opponent first, 0 tie / neither
        out["race_margin"] = np.array(margin)
        out["race_decided_at_root"] = root.get("race_root", [None, None]) != [None, None]
    out["n_distinct"] = len({r["pos_hash"] for r in rows})
    out["terminal"] = int(sum(r["terminal"] for r in rows))
    out["capped"] = int(sum(r["capped"] for r in rows))
    return out


def report(raw_dir: Path) -> dict:
    roots = []
    for f in sorted(raw_dir.glob("*.json")):
        roots += json.loads(f.read_text())["roots"]
    res = {"n_roots": len(roots), "per": {}}
    for w in WINDOWS:
        per_bucket = {}
        for bi, bname in enumerate(BUCKET_NAMES + ("all",)):
            sel = [r for r in roots if bname == "all" or bucket_of(r["t_s"]) == bi]
            if not sel:
                continue
            ms = [root_metrics(r, w) for r in sel]
            per_bucket[bname] = summarize(ms, sel)
        res["per"][f"W{w // DPS}s"] = per_bucket
    return res


def summarize(ms: list[dict], roots: list[dict]) -> dict:
    preds = sorted({n for m in ms for (n, _) in m["pred_pairs"]})
    s = {"roots": len(ms), "concordance": {}, "spearman": {}, "regret": {}, "gain_vs_no_search": {},
         "regret_win": {}}
    for t in TARGETS:
        gaps = np.concatenate([m["pairs"][t]["gap"] for m in ms])
        ses = np.concatenate([m["pairs"][t]["se"] for m in ms])
        s[f"pairs_{t}"] = {"n": int(len(gaps)), "abs_gap_median": float(np.median(np.abs(gaps))),
                           "se_median": float(np.median(ses)),
                           "frac_gap_gt_2se": float(np.mean(np.abs(gaps) > 2 * ses + 1e-9)),
                           "frac_gap_zero": float(np.mean(np.abs(gaps) < 1e-9))}
        for n in preds:
            sign = [m["pred_pairs"].get((n, t)) for m in ms]
            if any(x is None for x in sign):
                continue
            sign = np.concatenate(sign)
            row = {}
            scale = 100.0 if t == "win" else 1.0  # win-rate gaps in percentage points
            for thr in GAP_THRESHOLDS:
                sel = np.abs(gaps) * scale > max(thr, 1e-9) if thr == 0 else np.abs(gaps) * scale >= thr
                if sel.sum() == 0:
                    continue
                agree = np.where(sign[sel] == 0, 0.5, (sign[sel] == np.sign(gaps[sel])).astype(float))
                row[f">={thr:g}"] = [float(agree.mean()), int(sel.sum())]
            sel = (np.abs(gaps) > 2 * ses) & (np.abs(gaps) > 1e-9)
            if sel.sum():
                agree = np.where(sign[sel] == 0, 0.5, (sign[sel] == np.sign(gaps[sel])).astype(float))
                row[">2se"] = [float(agree.mean()), int(sel.sum())]
            s["concordance"].setdefault(t, {})[n] = row
            sp = np.array([m["spearman"].get((n, t), np.nan) for m in ms])
            s["spearman"].setdefault(t, {})[n] = [float(np.nanmean(sp)) if np.isfinite(sp).any() else None,
                                                  int(np.isfinite(sp).sum())]
    oracle = np.array([m["oracle"] for m in ms])
    oracle_cv = np.array([m["oracle_cv"] for m in ms])
    nos = np.array([m["no_search"] for m in ms])
    for n in sorted({k for m in ms for k in m["pick"]}):
        v = np.array([m["pick"].get(n, np.nan) for m in ms])
        s["regret"][n] = float(np.nanmean(oracle - v))
        s["gain_vs_no_search"][n] = [float(np.nanmean(v - nos)), float(np.nanstd(v - nos) / np.sqrt(np.isfinite(v).sum()))]
    for n in sorted({k for m in ms for k in m.get("pick_win", {})}):
        win = np.array([m["pick_win"].get(n, np.nan) for m in ms if "pick_win" in m])
        s["regret_win"][n] = float(np.nanmean(win))
    s["regret"]["random"] = float(np.mean(oracle - np.array([m["random"] for m in ms])))
    s["regret"]["no_search"] = float(np.mean(oracle - nos))
    s["regret"]["oracle_cv"] = float(np.mean(oracle - oracle_cv))
    s["gain_vs_no_search"]["random"] = [float(np.mean(np.array([m["random"] for m in ms]) - nos)), None]
    s["gain_vs_no_search"]["oracle"] = [float(np.mean(oracle - nos)), float(np.std(oracle - nos) / np.sqrt(len(ms)))]
    s["gain_vs_no_search"]["oracle_cv"] = [float(np.mean(oracle_cv - nos)), float(np.std(oracle_cv - nos) / np.sqrt(len(ms)))]
    s["headroom"] = {"oracle_ne_v23_frac": float(np.mean(oracle - nos > 1e-9)),
                     "oracle_gt_v23_by_1": float(np.mean(oracle - nos >= 1)),
                     "oracle_gt_v23_by_5": float(np.mean(oracle - nos >= 5)),
                     "oracle_cv_minus_v23_median": float(np.median(oracle_cv - nos)),
                     "best_branch_counts": _best_counts(ms)}
    s["distinct_end_states_mean"] = float(np.mean([m["n_distinct"] for m in ms]))
    s["terminal_branch_frac"] = float(np.mean([m["terminal"] for m in ms]) / len(BRANCHES))
    s["capped_rollouts"] = int(sum(m["capped"] for m in ms))
    # V's bias on the final lead (GT - V) of the V23 branch: is V calibrated for this continuation?
    s["v23_final_gt_mean"] = float(np.mean(nos))
    s["calib_v23_branch"] = {n: {"bias": float(np.mean([m["calib"][n] for m in ms])),
                                 "mae": float(np.mean([abs(m["calib"][n]) for m in ms]))} for n in ms[0]["calib"]}
    if "race_margin" in ms[0]:
        # pairs whose branches end the race differently vs the same, and V's concordance on each
        diff_gap, same_gap, conc = [], [], {"v_k1": [[], []], "sym_k10": [[], []], "hand_mlp": [[], []]}
        open_roots = 0
        for m in ms:
            rm = m["race_margin"]
            B = len(rm)
            iu = np.triu_indices(B, 1)
            differ = rm[iu[0]] != rm[iu[1]]
            open_roots += int(differ.any())
            gap = m["pairs"]["final"]["gap"]
            diff_gap += list(np.abs(gap[differ]))
            same_gap += list(np.abs(gap[~differ]))
            for n in conc:
                sign = m["pred_pairs"][(n, "final")]
                ok = np.abs(gap) > 1e-9
                agree = np.where(sign == 0, 0.5, (sign == np.sign(gap)).astype(float))
                conc[n][0] += list(agree[ok & differ])
                conc[n][1] += list(agree[ok & ~differ])
        s["race"] = {"roots_with_race_split": open_roots,
                     "pairs_race_differs": len(diff_gap), "abs_gap_race_differs": float(np.mean(diff_gap)) if diff_gap else None,
                     "pairs_race_same": len(same_gap), "abs_gap_race_same": float(np.mean(same_gap)) if same_gap else None,
                     "concordance_race_differs": {n: float(np.mean(v[0])) if v[0] else None for n, v in conc.items()},
                     "concordance_race_same": {n: float(np.mean(v[1])) if v[1] else None for n, v in conc.items()}}
    return s


def _best_counts(ms):
    counts = {}
    for m in ms:
        f = m["gt"]["final"]
        for b in np.flatnonzero(f >= f.max() - 1e-9):
            counts[BRANCHES[b]] = counts.get(BRANCHES[b], 0) + 1.0 / len(np.flatnonzero(f >= f.max() - 1e-9))
    return {k: round(v, 1) for k, v in sorted(counts.items(), key=lambda kv: -kv[1])}


# ------------------------------------------------------------------------------ driver


def _warm():
    from blackout_env.sim import SimState

    s = SimState.from_seed(1)
    s.step(np.zeros((10, 2), np.float32))
    s.obs()


def cmd_matches(args):
    _warm()
    tasks = match_tasks(args.seeds, args.seed0)
    out, started = [], time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        for i, fut in enumerate(as_completed([pool.submit(_match_worker, t) for t in tasks])):
            out.append(fut.result())
            if i % 50 == 0:
                print(f"matches {i}/{len(tasks)} {time.time() - started:.0f}s", flush=True)
    out.sort(key=lambda m: m["id"])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "matches.json").write_text(json.dumps(out, indent=0))
    lens = np.array([m["ticks"] for m in out]) / DPS
    print(f"{len(out)} matches, length median {np.median(lens):.1f}s, >30s {np.mean(lens > 30):.2f}, "
          f">60s {np.mean(lens > 60):.2f}", flush=True)


def cmd_branches(args):
    _warm()
    matches = json.loads((args.out / "matches.json").read_text())
    chosen = select_roots(matches, args.per_bucket, seed=args.select_seed)
    raw = args.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    by_id = {m["id"]: m for m in matches}
    jobs = []
    for mid, ticks in sorted(chosen.items()):
        if (raw / f"{mid}.json").exists():
            continue
        task = {k: by_id[mid][k] for k in ("id", "seed", "learner_team", "opponent")}
        jobs.append({"task": task, "root_ticks": sorted(ticks), "n_rollouts": args.rollouts})
    # longest first (late roots replay further and their matches run longer)
    jobs.sort(key=lambda j: -max(j["root_ticks"]))
    print(f"{sum(len(t) for t in chosen.values())} roots in {len(chosen)} matches; {len(jobs)} jobs to run", flush=True)
    started = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_init_branch_worker,
                             initargs=(args.value_ckpt, args.baselines)) as pool:
        futs = {pool.submit(_branch_worker, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs)):
            res = fut.result()
            (raw / f"{res['task']['id']}.json").write_text(json.dumps(res))
            print(f"[{i + 1}/{len(jobs)}] {res['task']['id']} {len(res['roots'])} roots {res['seconds']:.0f}s "
                  f"(elapsed {time.time() - started:.0f}s)", flush=True)


def cmd_report(args):
    res = report(args.out / "raw")
    (args.out / "report.json").write_text(json.dumps(res, indent=1))
    print(format_report(res))
    (args.out / "report.txt").write_text(format_report(res))


MAIN_PREDICTORS = ("lead", "hand_mlp", "v_k1", "v_k5", "v_k10", "sym_k1", "sym_k10", "lead_roll50", "v_roll50",
                   "sym_roll50", "lead_roll125", "v_roll125", "sym_roll125")


def format_report(res: dict) -> str:
    lines = [f"roots: {res['n_roots']}"]
    for wname, per in res["per"].items():
        for bname, s in per.items():
            lines.append(f"\n=== {wname} {bname}: {s['roots']} roots, distinct end states {s['distinct_end_states_mean']:.2f}/"
                         f"{len(BRANCHES)}, terminal {s['terminal_branch_frac']:.2f}, capped rollouts {s['capped_rollouts']}")
            pf = s["pairs_final"]
            lines.append(f"final-lead pairs {pf['n']}: |gap| median {pf['abs_gap_median']:.1f}, rollout SE median "
                         f"{pf['se_median']:.2f}, gap>2SE {pf['frac_gap_gt_2se']:.2f}, gap=0 {pf['frac_gap_zero']:.2f}")
            h = s["headroom"]
            lines.append(f"headroom: oracle != V23 {h['oracle_ne_v23_frac']:.2f}, >=1pt {h['oracle_gt_v23_by_1']:.2f}, "
                         f">=5pt {h['oracle_gt_v23_by_5']:.2f}; best: {h['best_branch_counts']}")
            lines.append(f"{'predictor':12s} {'conc>0':>12s} {'>=1':>12s} {'>=3':>12s} {'>=5':>12s} {'>2se':>12s} "
                         f"{'rho':>6s} {'regret':>7s} {'gain':>12s}")
            for n in MAIN_PREDICTORS + ("no_search", "random", "oracle_cv"):
                c = s["concordance"].get("final", {}).get(n, {})
                cells = [f"{c[k][0]:.3f}({c[k][1]})" if k in c else "-" for k in (">=0", ">=1", ">=3", ">=5", ">2se")]
                rho = s["spearman"].get("final", {}).get(n, [None])[0]
                g = s["gain_vs_no_search"].get(n)
                gs = "-" if g is None else (f"{g[0]:+.2f}" + (f"±{g[1]:.2f}" if g[1] is not None else ""))
                lines.append(f"{n:12s} " + " ".join(f"{x:>12s}" for x in cells)
                             + f" {('-' if rho is None else f'{rho:.3f}'):>6s} {s['regret'].get(n, float('nan')):7.2f} {gs:>12s}")
            lines.append("  regret picking by P(win) (final-lead pts): " + ", ".join(
                f"{n} {s['regret'][n + '@pwin']:.2f}" for n in ("v_k1", "sym_k1", "v_roll125", "sym_roll125")
                if n + "@pwin" in s["regret"]))
            lines.append("  calib (GT - pred, V23 branch): " + ", ".join(
                f"{n} {c['bias']:+.1f}/MAE {c['mae']:.1f}" for n, c in s["calib_v23_branch"].items()))
            if "race" in s:
                r = s["race"]
                lines.append(f"  race: roots with split race {r['roots_with_race_split']}, pairs differ {r['pairs_race_differs']} "
                             f"(|gap| {r['abs_gap_race_differs']}) vs same {r['pairs_race_same']} (|gap| {r['abs_gap_race_same']}); "
                             f"conc differ {r['concordance_race_differs']} same {r['concordance_race_same']}")
            for t in ("lead_h", "win"):
                c = s["concordance"].get(t, {})
                lines.append(f"  {t} concordance >0 / >=3 / >2se: " + ", ".join(
                    f"{n} {c[n].get('>=0', [np.nan])[0]:.3f}/{c[n].get('>=3', [np.nan])[0]:.3f}/{c[n].get('>2se', [np.nan])[0]:.3f}"
                    for n in ("lead", "hand_mlp", "v_k1", "v_k10", "sym_k1", "sym_k10") if n in c))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("matches", "branches", "report", "all"))
    p.add_argument("--out", type=Path, default=REPO / "reports/value_s1/branch")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--seeds", type=int, default=330, help="root-match seeds (x2 sides)")
    p.add_argument("--seed0", type=int, default=950000)
    p.add_argument("--per-bucket", type=int, default=120)
    p.add_argument("--select-seed", type=int, default=0)
    p.add_argument("--rollouts", type=int, default=4)
    p.add_argument("--value-ckpt", default=VALUE_CKPT)
    p.add_argument("--baselines", default=BASELINES)
    args = p.parse_args()
    os.chdir(REPO)
    if args.stage in ("matches", "all"):
        cmd_matches(args)
    if args.stage in ("branches", "all"):
        cmd_branches(args)
    if args.stage in ("report", "all"):
        cmd_report(args)


if __name__ == "__main__":
    main()
