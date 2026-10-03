"""S3 leaf check: does "5 s rollout from the branch-end state, then V" keep its branch-ranking
quality when the rollout policy is the small net r5 instead of strategic_v23?
(docs/policy_v2_design_ko.md §19 "S3 준비", follow-up of §15 S1.5 / §17 S1.6.)

Same 240 S1.5 roots x 8 branches, ground truth unchanged (S1.5's stored V23-continuation rollouts,
reports/value_s1/branch/raw).  Workers replay each root match deterministically (value_branch_eval
code paths, branch-end state hashes checked against S1.5), then per root:

  * ``v23``     leaf rollout learner fresh V23 + the true opponent (S1.5/S1.6's leaf, same reseed)
  * ``r5_true`` learner r5 (CPU, batched over the root's branches) + the true opponent
and return the branch-end states.  The main process then runs, batched over all ~1900 branch
ends on MPS fp32:
  * ``r5``      r5 for both teams (the S3 setting: the opponent is unknown), action repeat 1
  * ``r5k2``    the same with action repeat k = 2 (both teams, past the opening)
  * ``p11``     P11 student for both teams (MPS fp16), a stronger / slower net rollout
and, as references (CPU workers), ``v23_r5opp`` = learner V23 + r5 opponent.
each to 3 s (75 decisions) and 5 s (125), common reseed per root (= S1.5's leaf reseed).
V in {s2v_iter5, s2v_iter5 symmetric, s1b symmetric} (+ the lead alone) is read at the leaf end;
leaf rollouts that end the match are scored exactly.

    PYTHONPATH=. .venv/bin/python examples/leaf_rollout_eval.py --workers 12 --out reports/leaf_rollout

Value-net comparison (docs §27): only the S3 leaf (r5 x r5, 5 s) and the r5 x r5 ground truth,
generated once and cached, then any number of V checkpoints scored on the cache:

    ... leaf_rollout_eval.py --r5-only --gt-r5 --cache <dir>/leaf_cache.pkl --values a=path b=path --out <dir>
    ... leaf_rollout_eval.py --from-cache <dir>/leaf_cache.pkl --values c=path --out <dir>
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "examples"))

R5 = "checkpoints/rollout_net/r5.pt"
VALUES = {"iter5": "checkpoints/value/frozen/s2v_iter5.pt", "s1b": "checkpoints/value/frozen/s1b_value.pt"}
LEAF_RESEED = 7_777_777  # value_branch_eval.run_root's leaf rollout reseed (+ rng_seed)
HORIZONS = (75, 125)     # 3 s, 5 s
W = 50

# ------------------------------------------------------------------------------ worker


class InstancePolicy:
    """BatchPolicy over per-game policy instances (``act`` on env obs dicts), one team per instance."""

    needs_inputs = False

    def __init__(self, instances: dict):
        self.instances = instances  # game index -> policy (already bound to its team)

    def reset(self, n):
        pass

    def moves(self, bsim, idx, teams, inputs):
        from blackout_env.sim.state import obs_from_raw

        out = {tm: np.zeros((len(idx), 5, 2), np.float32) for tm in teams}
        for j, b in enumerate(idx):
            obs = obs_from_raw(*bsim.raw(int(b)))
            for tm in teams:
                names = [f"unit_{i}" for i in range(5 * tm, 5 * tm + 5)]
                acts = self.instances[int(b)].act({n: obs[n] for n in names})
                for n_, a in acts.items():
                    out[tm][j, int(n_.split("_")[1]) - 5 * tm] = np.clip(np.asarray(a, np.float32), -1, 1)
        return out


_R5 = None


def _leaf_record(sim, team, encode):
    """Leaf end -> ('term', final, win) or ('view', own, mir, lead)."""
    import value_branch_eval as vb
    from blackout_env.value.evaluate import encode_view, team_view

    if sim.over:
        return ("term", vb.lead_of(sim, team), vb.result_of(sim, team))
    obs = sim.obs()
    return ("view", encode_view(team_view(obs, team)), encode_view(team_view(obs, 1 - team)), vb.lead_of(sim, team))


def _worker(job):
    global _R5
    import torch

    import value_branch_eval as vb
    from blackout_env.sim.batch import BatchRollout, NetPolicy
    from blackout_env.sim.rollout_net import RolloutNet

    torch.set_num_threads(1)
    if _R5 is None:
        _R5 = NetPolicy(RolloutNet.from_checkpoint(str(REPO / R5)), device="cpu")
    task, root_ticks = job["task"], job["root_ticks"]
    team, desc = task["learner_team"], task["opponent"]
    snaps = {}

    def on_tick(tick, sim, opp, shadow):
        if tick in root_ticks:
            snaps[tick] = (sim.copy(), vb.clone_opponent(desc, opp), {k: copy.deepcopy(v) for k, v in shadow.items()})

    vb.play_root_match(task, stop_tick=max(root_ticks) + 1, on_tick=on_tick, shadows=True)
    out = []
    for tick in root_ticks:
        sim_root, opp_root, shadows = snaps[tick]
        rng_seed = zlib.crc32(f"{task['id']}:{tick}".encode()) % 1_000_000
        ends = []  # (sim_w, opp_w, reached)
        for b, name in enumerate(vb.BRANCHES):  # == value_branch_eval.run_root's window loop
            pol = vb.branch_policy(name, shadows, rng_seed * 31 + b, team)
            opp = vb.clone_opponent(desc, opp_root)
            sim = sim_root.copy()
            step, end = 0, None
            while step < W and not sim.over:
                sim.step(vb.joint_moves(sim.obs(), pol, opp, team))
                step += 1
                if step == W:
                    end = (sim.copy(), vb.clone_opponent(desc, opp), step)
            ends.append(end if end is not None else (sim.copy(), opp, step))
        hashes = [int(zlib.crc32(s.uf.tobytes() + s.ui.tobytes())) for s, _, _ in ends]
        live = [b for b, (s, _, _) in enumerate(ends) if not s.over]
        rec = {"rid": f"{task['id']}@{tick}", "team": team, "hashes": hashes, "live": live, "leaf": {}, "time": {}}
        if job.get("r5_only"):
            rec["sims"] = pickle.dumps([ends[b][0] for b in live])
            rec["reseed"] = LEAF_RESEED + rng_seed
            out.append(rec)
            continue
        # V23 leaf (S1.5 / S1.6): learner fresh V23 + true opponent, snapshots at 3 s / 5 s
        t0 = time.perf_counter()
        for h in HORIZONS:
            rec["leaf"][f"v23_true@{h}"] = {}
        for b in live:
            s, o, reached = ends[b]
            c = vb.continuation(s, desc, o, team, LEAF_RESEED + rng_seed, tick + reached, max(HORIZONS),
                                snap_at=set(HORIZONS))
            for h in HORIZONS:
                if h in c["snaps"]:
                    from blackout_env.value.evaluate import encode_view, team_view

                    obs = c["snaps"][h]
                    rec["leaf"][f"v23_true@{h}"][b] = ("view", encode_view(team_view(obs, team)),
                                                      encode_view(team_view(obs, 1 - team)), None)
                else:
                    rec["leaf"][f"v23_true@{h}"][b] = ("term", c["final"], c["win"])
        rec["time"]["v23_true"] = time.perf_counter() - t0
        # r5 learner + true opponent (CPU, batched over this root's live branches)
        t0 = time.perf_counter()
        if live:
            for h in HORIZONS:
                opps = {j: vb.clone_opponent(desc, ends[b][1]) for j, b in enumerate(live)}
                pa, pb = (_R5, InstancePolicy(opps)) if team == 0 else (InstancePolicy(opps), _R5)
                res = BatchRollout(pa, pb, repeat=1).run([ends[b][0] for b in live], h,
                                                         reseeds=[LEAF_RESEED + rng_seed] * len(live))
                rec["leaf"][f"r5_true@{h}"] = {b: _leaf_record(res.bsim.state(j), team, None) for j, b in enumerate(live)}
        rec["time"]["r5_true"] = time.perf_counter() - t0
        # reference: learner fresh V23 (= the ground truth's continuation policy) + r5 opponent
        t0 = time.perf_counter()
        if live:
            for h in HORIZONS:
                v23s = {j: vb._make("strategic_v23") for j in range(len(live))}
                pa, pb = (InstancePolicy(v23s), _R5) if team == 0 else (_R5, InstancePolicy(v23s))
                res = BatchRollout(pa, pb, repeat=1).run([ends[b][0] for b in live], h,
                                                         reseeds=[LEAF_RESEED + rng_seed] * len(live))
                rec["leaf"][f"v23_r5opp@{h}"] = {b: _leaf_record(res.bsim.state(j), team, None) for j, b in enumerate(live)}
        rec["time"]["v23_r5opp"] = time.perf_counter() - t0
        rec["sims"] = pickle.dumps([ends[b][0] for b in live])
        rec["reseed"] = LEAF_RESEED + rng_seed
        out.append(rec)
    return out


# ------------------------------------------------------------------------------ main: batched r5 x r5


P11 = "checkpoints/distill/frozen/p11_v22.pt"


def run_r5_self(recs, repeat: int, horizon: int, device: str, net: str = "r5"):
    """Both teams the same net (r5 MPS fp32, or P11 MPS fp16), batched over every live branch end."""
    import torch

    from blackout_env.sim.batch import BatchRollout, NetPolicy
    from blackout_env.sim.rollout_net import RolloutNet

    if net == "r5":
        pol = NetPolicy(RolloutNet.from_checkpoint(str(REPO / R5)), device=device)
    else:
        pol = NetPolicy.from_checkpoint(str(REPO / P11), device=device, fp16=True)
    sims, reseeds, where = [], [], []
    for r in recs:
        ss = pickle.loads(r["sims"])
        for b, s in zip(r["live"], ss):
            sims.append(s)
            reseeds.append(r["reseed"])
            where.append((r, b))
    roll = BatchRollout(pol, pol, repeat=repeat)
    roll.run(sims[:64], 5, reseeds=reseeds[:64])  # warm-up (MPS kernels)
    if device == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    res = roll.run(sims, horizon, reseeds=reseeds)
    t_roll = time.perf_counter() - t0
    t0 = time.perf_counter()
    key = f"{net}{'k%d' % repeat if repeat > 1 else ''}@{horizon}"
    for j, (r, b) in enumerate(where):
        r["leaf"].setdefault(key, {})[b] = _leaf_record(res.bsim.state(j), r["team"], None)
    t_enc = time.perf_counter() - t0
    return key, {"n": len(sims), "rollout_s": t_roll, "encode_s": t_enc, "timing": roll.timing}


def value_preds(recs, device: str, values: dict | None = None) -> dict:
    """(variant, rid, b) -> {net: (own_final, mir_final)} in points, + lead; one batched pass per net."""
    from blackout_env.value.evaluate import ValueEvaluator

    keys, own, mir = [], [], []
    for r in recs:
        for var, d in r["leaf"].items():
            for b, x in d.items():
                if x[0] == "view":
                    keys.append((var, r["rid"], b))
                    own.append(x[1])
                    mir.append(x[2])
    stack = lambda es: [np.stack([e[i] for e in es]) for i in range(3)]  # noqa: E731
    o, m = stack(own), stack(mir)
    out, timing = {k: {} for k in keys}, {}
    for name, path in (values or VALUES).items():
        ev = ValueEvaluator(str(REPO / path), device=device, batch=1024)
        ev.evaluate_encoded(*[x[:256] for x in o])  # warm-up
        t0 = time.perf_counter()
        po = ev.evaluate_encoded(*o)
        timing[name] = (time.perf_counter() - t0) / len(keys)
        pm = ev.evaluate_encoded(*m)
        for i, k in enumerate(keys):
            out[k][name] = (float(po["final"][i] * 100), float(pm["final"][i] * 100))
            out[k]["lead"] = float(po["lead"][i] * 100)
    return out, timing


# ------------------------------------------------------------------------------ metrics

PREDICTORS = ("lead", "iter5", "iter5_sym", "s1b_sym")


def predictors_for(values: dict) -> tuple:
    return ("lead",) + tuple(x for n in values for x in (n, f"{n}_sym"))


def pred_vector(root, rec, var, V, predictor) -> np.ndarray:
    B = len(root["branches"])
    v = np.full(B, np.nan)
    for b, br in enumerate(root["branches"]):
        row = br["windows"][str(W)]
        if row["terminal"]:
            v[b] = row["final_r"][0]
            continue
        x = rec["leaf"][var][b]
        if x[0] == "term":
            v[b] = x[1]
            continue
        p = V[(var, rec["rid"], b)]
        if predictor == "lead":
            v[b] = p["lead"]
        elif predictor.endswith("_sym"):
            o, m = p[predictor[:-4]]
            v[b] = (o - m) / 2
        else:
            v[b] = p[predictor][0]
    return v


def leaf_calibration(roots, recs, V, var, predictor, mkey="_m_r5") -> dict:
    """Mean (V at the leaf - ground-truth final of the branch) over non-terminal leaves, by bucket
    (the GT continuation starts at the branch end, the leaf 5 s later on another reseed of the same
    policy pair, so a calibrated V has zero mean error); also the root-centred error spread."""
    import value_branch_eval as vb

    out = {}
    for bi, bname in enumerate(vb.BUCKET_NAMES + ("all",)):
        err, cerr = [], []
        for root in roots:
            if bname != "all" and vb.bucket_of(root["t_s"]) != bi:
                continue
            rec = recs[f"{root['match']}@{root['tick']}"]
            gt = root[mkey]["gt"]["final"]
            pv = pred_vector(root, rec, var, V, predictor)
            live = [b for b in range(len(gt)) if (var, rec["rid"], b) in V]
            if not live:
                continue
            e = pv[live] - gt[live]
            err += list(e)
            if len(live) >= 2:
                cerr += list(e - e.mean())
        out[bname] = {"n": len(err), "bias": float(np.mean(err)) if err else float("nan"),
                      "mae": float(np.mean(np.abs(err))) if err else float("nan"),
                      "centred_mae": float(np.mean(np.abs(cerr))) if cerr else float("nan")}
    return out


def gt_r5(recs, roots, device: str, R: int = 4, cap: int = 1500):
    """Alternative ground truth: the branch-end states continued by r5 for both teams (MPS), R
    reseeds (S1.5's 1_000_003 (r + 1) + rng_seed, common across branches), to the end or 60 s."""
    from blackout_env.sim.batch import BatchRollout, NetPolicy
    from blackout_env.sim.rollout_net import RolloutNet

    pol = NetPolicy(RolloutNet.from_checkpoint(str(REPO / R5)), device=device)
    sims, seeds, where = [], [], []
    for r in recs.values():
        rng_seed = r["reseed"] - LEAF_RESEED
        ss = pickle.loads(r["sims"])
        for b, s in zip(r["live"], ss):
            for k in range(R):
                sims.append(s)
                seeds.append(1_000_003 * (k + 1) + rng_seed)
                where.append((r, b, k))
    t0 = time.perf_counter()
    res = BatchRollout(pol, pol).run(sims, cap, reseeds=seeds)
    for j, (r, b, k) in enumerate(where):
        r.setdefault("gt_r5", {}).setdefault(b, [0] * R)[k] = float(res.lead(r["team"])[j])
    capped = float(np.mean(~res.over))
    for root in roots:
        rec = recs[f"{root['match']}@{root['tick']}"]
        rows = []
        for b, br in enumerate(root["branches"]):
            row = br["windows"][str(W)]
            rows.append([row["final_r"][0]] * R if row["terminal"] else rec["gt_r5"][b])
        g = np.array(rows, float)
        iu = np.triu_indices(len(g), 1)
        h = R // 2
        a, c = g[:, :h].mean(1), g[:, h:].mean(1)
        import value_branch_eval as vb

        f = g.mean(1)
        root["_m_r5"] = {"gt": {"final": f}, "pairs": {"final": {"gap": (g[iu[0]] - g[iu[1]]).mean(1)}},
                         "no_search": float(f[vb.BRANCHES.index("v23")]), "oracle": float(f.max()),
                         "oracle_cv": 0.5 * (vb._ties_argmax_mean(a, c) + vb._ties_argmax_mean(c, a))}
    return {"n": len(sims), "s": time.perf_counter() - t0, "capped_frac": capped}


def metrics(roots, recs, V, var, predictor, mkey="_m") -> dict:
    import value_branch_eval as vb

    agree, agree5, gains, regrets, ocv = [], [], [], [], []
    for root in roots:
        rec = recs[f"{root['match']}@{root['tick']}"]
        m = root[mkey]
        gt = m["gt"]["final"]
        pv = pred_vector(root, rec, var, V, predictor)
        iu = np.triu_indices(len(gt), 1)
        gap = m["pairs"]["final"]["gap"]
        sign = np.sign(pv[iu[0]] - pv[iu[1]])
        ok = np.abs(gap) > 1e-9
        a = np.where(sign == 0, 0.5, (sign == np.sign(gap)).astype(float))
        agree += list(a[ok])
        agree5 += list(a[np.abs(gap) >= 5])
        pick = vb._ties_argmax_mean(pv, gt)
        gains.append(pick - m["no_search"])
        regrets.append(m["oracle"] - pick)
        ocv.append(m["oracle_cv"] - m["no_search"])
    g = np.array(gains)
    return {"conc": float(np.mean(agree)), "conc5": float(np.mean(agree5)), "gain": float(g.mean()),
            "gain_se": float(g.std() / np.sqrt(len(g))), "regret": float(np.mean(regrets)),
            "oracle_cv": float(np.mean(ocv)), "share": float(g.mean() / np.mean(ocv)), "n": len(g)}


# ------------------------------------------------------------------------------ driver


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--s15", type=Path, default=REPO / "reports/value_s1/branch")
    p.add_argument("--out", type=Path, default=REPO / "reports/leaf_rollout")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--device", default="mps")
    p.add_argument("--gt-r5", action="store_true", help="also score against an r5 x r5 continuation ground truth")
    p.add_argument("--limit", type=int, default=0, help="first N matches only (smoke test)")
    p.add_argument("--r5-only", action="store_true", help="only the r5 x r5 5 s leaf (+ --gt-r5): no V23 / P11 leaves")
    p.add_argument("--values", nargs="+", default=None, help="name=path value nets to score (default iter5, s1b)")
    p.add_argument("--cache", type=Path, default=None, help="save leaf states + ground truth here (pickle)")
    p.add_argument("--from-cache", type=Path, default=None, help="score --values on a saved cache")
    p.add_argument("--tag", default="", help="results_<tag>.json/.txt")
    args = p.parse_args()
    import os

    os.chdir(REPO)
    import value_branch_eval as vb

    values = dict(v.split("=", 1) for v in args.values) if args.values else dict(VALUES)
    if args.from_cache:
        with open(args.from_cache, "rb") as f:
            c = pickle.load(f)
        return score(c["roots"], c["recs"], c["timing"], values, args, gt=c["gt_r5"])
    vb._warm()
    files = sorted((args.s15 / "raw").glob("*.json"))
    raws = [json.loads(f.read_text()) for f in files]
    if args.limit:
        raws = raws[: args.limit]
    roots = [r for x in raws for r in x["roots"]]
    jobs = [{"task": x["task"], "root_ticks": sorted(r["tick"] for r in x["roots"]), "r5_only": args.r5_only}
            for x in raws]
    print(f"{len(jobs)} matches, {len(roots)} roots", flush=True)
    t0 = time.time()
    recs = {}
    with ProcessPoolExecutor(args.workers) as pool:
        futs = [pool.submit(_worker, j) for j in sorted(jobs, key=lambda j: -max(j["root_ticks"]))]
        for i, fut in enumerate(as_completed(futs)):
            for rec in fut.result():
                recs[rec["rid"]] = rec
            if i % 20 == 0:
                print(f"[{i + 1}/{len(jobs)}] {time.time() - t0:.0f}s", flush=True)
    t_workers = time.time() - t0
    bad = sum(int(recs[f"{r['match']}@{r['tick']}"]["hashes"] != [br["windows"][str(W)]["pos_hash"] for br in r["branches"]])
              for r in roots)
    print(f"workers {t_workers:.0f}s; branch-end hash mismatches {bad}/{len(roots)}", flush=True)
    n_live = sum(len(r["live"]) for r in recs.values())
    timing = {"workers_wall_s": t_workers, "hash_mismatch_roots": bad, "live_leaves": n_live}
    if not args.r5_only:
        timing.update({
            "v23_true_cpu_s_per_leaf": sum(r["time"]["v23_true"] for r in recs.values()) / n_live,
            "r5_true_cpu_s_per_leaf_3s_plus_5s": sum(r["time"]["r5_true"] for r in recs.values()) / n_live,
            "v23_r5opp_cpu_s_per_leaf_3s_plus_5s": sum(r["time"]["v23_r5opp"] for r in recs.values()) / n_live})
    rl = list(recs.values())
    for rep, h, net in ([(1, 125, "r5")] if args.r5_only else
                        [(rep, h, "r5") for rep in (1, 2) for h in HORIZONS] + [(1, h, "p11") for h in HORIZONS]):
        key, t = run_r5_self(rl, rep, h, args.device, net=net)
        timing[key] = {**t, "ms_per_leaf_rollout": 1000 * t["rollout_s"] / t["n"]}
        print(key, json.dumps(timing[key]), flush=True)
    for r in roots:
        r["_m"] = vb.root_metrics(r, W)
    if args.gt_r5:
        timing["gt_r5"] = gt_r5(recs, roots, args.device)
        print("gt_r5", timing["gt_r5"], flush=True)
    if args.cache:
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        with open(args.cache, "wb") as f:
            pickle.dump({"roots": roots, "recs": recs, "timing": timing, "gt_r5": bool(args.gt_r5)}, f)
        print(f"cache -> {args.cache}", flush=True)
    score(roots, recs, timing, values, args, gt=args.gt_r5)


def score(roots, recs, timing, values: dict, args, gt: bool):
    import value_branch_eval as vb

    rl = list(recs.values())
    V, vt = value_preds(rl, args.device, values)
    timing = {**timing, "value_ms_per_view": {k: 1000 * v for k, v in vt.items()}}
    PREDICTORS = predictors_for(values)  # noqa: N806
    variants = sorted({k for rec in rl for k in rec["leaf"]})
    buckets = vb.BUCKET_NAMES + ("all",)
    table = {}
    for var in variants:
        for pr in PREDICTORS:
            for bi, bname in enumerate(buckets):
                sel = [r for r in roots if bname == "all" or vb.bucket_of(r["t_s"]) == bi]
                table[f"{var}|{pr}|{bname}"] = metrics(sel, recs, V, var, pr)
                if gt:
                    table[f"gtr5|{var}|{pr}|{bname}"] = metrics(sel, recs, V, var, pr, "_m_r5")
    calib = {}
    if gt:
        for var in variants:
            for pr in PREDICTORS[1:]:
                calib[f"{var}|{pr}"] = leaf_calibration(roots, recs, V, var, pr)
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    (args.out / f"results{tag}.json").write_text(json.dumps({"timing": timing, "table": table, "leaf_calib_gtr5": calib},
                                                           indent=1))
    lines = [json.dumps(timing, indent=1, default=float), "",
             f"{'variant':14s} {'V':10s} " + " ".join(f"{b:>22s}" for b in buckets)]
    for var in variants:
        for pr in PREDICTORS:
            cells = []
            for bname in buckets:
                m = table[f"{var}|{pr}|{bname}"]
                cells.append(f"{m['conc']:.3f} {m['gain']:+5.2f}±{m['gain_se']:.1f} {100 * m['share']:3.0f}%")
            lines.append(f"{var:14s} {pr:10s} " + " ".join(f"{c:>22s}" for c in cells))
    if gt:
        lines.append("\n--- ground truth = r5 x r5 continuation (R=4, cap 60 s)")
        for var in variants:
            for pr in PREDICTORS:
                cells = []
                for bname in buckets:
                    m = table[f"gtr5|{var}|{pr}|{bname}"]
                    cells.append(f"{m['conc']:.3f} {m['gain']:+5.2f}±{m['gain_se']:.1f} {100 * m['share']:3.0f}%")
                lines.append(f"{var:14s} {pr:10s} " + " ".join(f"{c:>22s}" for c in cells))
        lines.append("oracle_cv gain (r5 GT): " + ", ".join(
            f"{b} {table[f'gtr5|{variants[0]}|lead|{b}']['oracle_cv']:+.2f}" for b in buckets))
        lines.append("\n--- leaf calibration vs r5 x r5 GT: bias / MAE / root-centred MAE (pts)")
        for k, c in calib.items():
            lines.append(f"{k:24s} " + " ".join(f"{b}: {c[b]['bias']:+5.1f}/{c[b]['mae']:4.1f}/{c[b]['centred_mae']:4.1f}"
                                                 for b in buckets))
    lines.append("oracle_cv gain: " + ", ".join(f"{b} {table[f'{variants[0]}|lead|{b}']['oracle_cv']:+.2f}" for b in buckets))
    txt = "\n".join(lines)
    (args.out / f"results{tag}.txt").write_text(txt)
    print(txt, flush=True)


if __name__ == "__main__":
    main()
