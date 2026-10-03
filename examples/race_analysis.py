"""H9: what does the S3 search change in the opening race to the Hunter sanctuary?
(docs/policy_v2_design_ko.md §28 "탐색의 경주 승리 분석".)  Simulator only.

``play``: for each (seed, side) the plain rollout net r5 and the race-only search (the §23
``race10`` setting: B8 ``alt*,samp:1.0x2``, every 25 decisions, window 25, margin 3, searching
while t < ``--search-until``) play the same match start against one opponent for ``--cap``
decisions.  Every decision stores both teams' unit positions and classes, the executed moves,
the branch the search team executes and the sim events (transforms, kills, pickups) -> one npz
per tag: ``<out>/<tag>_<opponent>.npz``.

``analyze``: pairs plain vs search per (seed, side): race (first Hunter tick) outcome of both,
first deviation (tick, unit, branch), which unit transforms first, per-walker sanctuary entry
ticks, kills before / after the race, the opponent's first reaction, and a mechanism class for
every match whose race outcome the search improves.  Writes ``<out>/analysis_<opponent>.json``
and prints a summary table.

    PYTHONPATH=. .venv/bin/python examples/race_analysis.py play --opponent strategic_v23 --seeds 128
    PYTHONPATH=. .venv/bin/python examples/race_analysis.py analyze --opponent strategic_v23
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

R5 = "checkpoints/rollout_net/r5.pt"
VALUE = "checkpoints/value/frozen/s2v_iter5.pt"
P11 = "checkpoints/distill/frozen/p11_v22.pt"
SANCT = (11.0, 13.0)  # Hunter Sanctuary tiles x, y in [11, 13)


def make_opponent(name: str, our, device: str):
    from blackout_env.sim.batch import HeuristicPolicy, NetPolicy

    if name == "r5":
        return our
    if name == "p11":
        return NetPolicy.from_checkpoint(str(REPO / P11), device=device)
    return HeuristicPolicy(name)


def run_tag(seeds, sides, net, opponent_name, device, searcher, search_until, cap, every=25, window=25):
    """One lockstep chunk; returns the recorded arrays."""
    from blackout_env.sim import core
    from blackout_env.sim.batch import N_AGENT_COLS, N_CH, BatchSim
    from blackout_env.sim.mapdata import HUNTER, H, W
    from blackout_env.sim.search import ModNetPolicy
    from blackout_env.sim.state import SimState

    F32 = np.float32
    n = len(seeds)
    sides = np.asarray(sides, np.int64)
    our = ModNetPolicy(net)
    opp = make_opponent(opponent_name, our, device)
    bs = BatchSim([SimState.from_seed(int(s)) for s in seeds])
    our.configure(n)
    if opp is not our:
        opp.reset(n)
    bufs = (np.zeros((n, 2, N_CH, H, W), F32), np.zeros((n, 2, 4), F32), np.zeros((n, 2, 10, N_AGENT_COLS), F32))
    moves = np.zeros((n, 10, 2), F32)
    pos = np.zeros((cap + 1, n, 10, 2), F32)
    cls = np.zeros((cap + 1, n, 10), np.int8)
    mv = np.zeros((cap, n, 10, 2), F32)
    branch = np.zeros((cap, n), np.int16)
    nb = len(searcher.branches) if searcher else 1
    svals = np.full((cap // every + 1, n, nb), np.nan, np.float32)
    events = []  # (game, tick, kind, a, b, c, d)
    cur = np.zeros(n, np.int64)
    pos[0], cls[0] = bs.uf, bs.ui[:, :, core.U_CLS]
    for t in range(cap):
        live = np.nonzero(~bs.over)[0]
        if not len(live):
            break
        if searcher is not None and t % every == 0 and t < search_until:
            roots = [bs.state(int(b)) for b in live]
            reseeds = [(int(seeds[b]) * 7919 + t * 104729 + 12345) % (2 ** 31) for b in live]
            out = searcher.search(roots, sides[live], reseeds)
            for j, b in enumerate(live):
                k = int(out.choice[j])
                our.set_game(b, sides[b], searcher.branches[k], window)
                cur[b] = k
            svals[t // every, live] = out.values
        bs.render(live, *bufs)
        sub = tuple(x[: len(live)] for x in bufs)
        ours = sides[live]
        branch[t, live] = np.where(our.left[live] > 0, cur[live], 0)
        mo = our.moves(bs, live, (0, 1), sub)
        if opp is our:
            oo = mo
        else:
            oo = {tm: np.zeros((len(live), 5, 2), F32) for tm in (0, 1)}
            for team in (0, 1):
                sel = ours != team
                if sel.any():
                    s_in = sub if sel.all() else tuple(x[sel] for x in sub)
                    oo[team][sel] = opp.moves(bs, live[sel], (team,), s_in)[team]
        for team in (0, 1):
            mine = ours == team
            moves[live, 5 * team:5 * team + 5] = np.where(mine[:, None, None], mo[team], oo[team])
        mv[t, live] = moves[live]
        bs.step(moves, live)
        for b in live:
            ne = int(bs.g[b, core.G_NEV])
            for e in bs.ev[b, :ne]:
                if e[1] in (core.EV_KILL, core.EV_TRANSFORM, core.EV_PICKUP):
                    events.append((int(b), t + 1, *map(int, e[1:])))
        pos[t + 1], cls[t + 1] = bs.uf, bs.ui[:, :, core.U_CLS]
    return {"seed": np.asarray(seeds), "side": sides, "pos": pos, "cls": cls, "moves": mv, "branch": branch,
            "svals": svals, "events": np.asarray(events, np.int64).reshape(-1, 7),
            "scores": bs.scores, "hunter": np.int64(HUNTER)}


def cmd_play(args):
    import torch

    from blackout_env.sim.batch import NetPolicy
    from blackout_env.sim.rollout_net import RolloutNet
    from blackout_env.sim.search import ModNetPolicy, Searcher, ValueNetLeaf, default_device, parse_branches

    torch.set_num_threads(args.threads)
    device = args.device or default_device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    net = NetPolicy(RolloutNet.from_checkpoint(str(REPO / R5)), device=device)
    branches = parse_branches(args.branches)
    searcher = Searcher(ModNetPolicy(net), ValueNetLeaf(str(REPO / VALUE), device), branches, window=25,
                        margin=args.margin)
    tasks = [(s, side) for s in range(args.seed0, args.seed0 + args.seeds) for side in (0, 1)]
    for tag, srch in (("plain", None), ("search", searcher)):
        path = out / f"{tag}_{args.opponent}.npz"
        if path.exists() and not args.force:
            print("exists", path)
            continue
        parts = []
        for c in range(0, len(tasks), args.chunk):
            chunk = tasks[c:c + args.chunk]
            t0 = time.time()
            parts.append(run_tag([s for s, _ in chunk], [sd for _, sd in chunk], net, args.opponent, device, srch,
                                 args.search_until, args.cap))
            print(f"[{tag} vs {args.opponent}] {c + len(chunk)}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
        merged = {}
        off = 0
        for p in parts:
            ev = p["events"].copy()
            ev[:, 0] += off
            p["events"] = ev
            off += len(p["seed"])
        for k in parts[0]:
            if k == "hunter":
                merged[k] = parts[0][k]
            elif k in ("seed", "side", "scores", "events"):
                merged[k] = np.concatenate([p[k] for p in parts])
            else:
                merged[k] = np.concatenate([p[k] for p in parts], axis=1)
        merged["branches"] = np.asarray(json.dumps([b.name for b in (branches if srch else branches[:1])]))
        np.savez_compressed(path, **merged)
        print("wrote", path)


# ---------------------------------------------------------------------------------- analysis

def first_hunter(cls, team, hunter):
    """[T+1, n, 10] -> [n] first tick with a Hunter on ``team`` (big if none)."""
    h = (cls[:, :, 5 * team:5 * team + 5] == hunter).any(-1)
    any_ = h.any(0)
    return np.where(any_, h.argmax(0), 10 ** 6)


def sanct_dist(p):
    """Euclidean distance (tiles) from points [..., 2] to the 2x2 sanctuary square."""
    dx = np.maximum(np.maximum(SANCT[0] - p[..., 0], p[..., 0] - SANCT[1]), 0)
    dy = np.maximum(np.maximum(SANCT[0] - p[..., 1], p[..., 1] - SANCT[1]), 0)
    return np.hypot(dx, dy)


def load(out: Path, tag: str, opponent: str):
    d = np.load(out / f"{tag}_{opponent}.npz")
    return {k: d[k] for k in d.files}


def game_events(ev, g):
    return ev[ev[:, 0] == g]


def analyze_game(P, S, g, hunter):
    """Paired plain (P) vs search (S) facts for game g."""
    side = int(P["side"][g])
    opp = 1 - side
    our_u = range(5 * side, 5 * side + 5)
    opp_u = range(5 * opp, 5 * opp + 5)
    r = {"seed": int(P["seed"][g]), "side": side}
    for tag, D in (("p", P), ("s", S)):
        fo = int(first_hunter(D["cls"][:, g:g + 1], side, hunter)[0])
        fp = int(first_hunter(D["cls"][:, g:g + 1], opp, hunter)[0])
        r[f"{tag}_first_ours"], r[f"{tag}_first_opp"] = fo, fp
        r[f"{tag}_race"] = int(np.sign(fp - fo))
        ev = game_events(D["events"], g)
        tr = ev[(ev[:, 2] == 3) & (ev[:, 4] == hunter)]  # EV_TRANSFORM to Hunter: (g, tick, kind, u, cls, reg, 0)
        r[f"{tag}_transforms"] = [(int(t), int(u)) for t, u in tr[:, [1, 3]][:8]]
        kills = ev[ev[:, 2] == 2]  # EV_KILL (g, tick, kind, killer, victim, vcls, kcls)
        r[f"{tag}_kills"] = [(int(t), int(k), int(v)) for t, k, v in kills[:, [1, 3, 4]] if t <= 300][:12]
        # first transforming unit of each team
        ft_o = [u for t, u in r[f"{tag}_transforms"] if u in our_u]
        ft_p = [u for t, u in r[f"{tag}_transforms"] if u in opp_u]
        r[f"{tag}_first_unit_ours"] = ft_o[0] if ft_o else -1
        r[f"{tag}_first_unit_opp"] = ft_p[0] if ft_p else -1
        # walkers' distance to the sanctuary just before the race decision
        r[f"{tag}_score"] = D["scores"][g].tolist()
    # first deviation of our moves / of the opponent's moves
    T = P["moves"].shape[0]
    dm = np.abs(P["moves"][:, g] - S["moves"][:, g]).max(-1)  # [T, 10]
    dour = dm[:, list(our_u)].max(-1) > 1e-6
    dopp = dm[:, list(opp_u)].max(-1) > 1e-6
    r["dev_tick"] = int(dour.argmax()) if dour.any() else -1
    r["opp_react_tick"] = int(dopp.argmax()) if dopp.any() else -1
    if r["dev_tick"] >= 0:
        t0 = r["dev_tick"]
        r["dev_units"] = [int(u) - 5 * side for u in our_u if dm[t0, u] > 1e-6]
        r["dev_branch"] = int(S["branch"][t0, g])
    # executed branches before the race decision
    tdec = min(r["s_first_ours"], r["s_first_opp"], T)
    br = S["branch"][:tdec, g]
    r["branches_pre_race"] = [int(x) for x in np.unique(br[br > 0])]
    r["nonbase_ticks_pre_race"] = int((br > 0).sum())
    return r


def mechanism(r, P, S, g, hunter):
    """Mechanism class of a match whose race outcome the search improved."""
    side = r["side"]
    opp = 1 - side
    our_u = list(range(5 * side, 5 * side + 5))
    opp_u = list(range(5 * opp, 5 * opp + 5))
    dfo = r["s_first_ours"] - r["p_first_ours"]
    dfp = r["s_first_opp"] - r["p_first_opp"]
    out = {"d_first_ours": dfo, "d_first_opp": dfp,
           "same_first_unit": r["s_first_unit_ours"] == r["p_first_unit_ours"]}
    # opp walker killed before transforming in the search game? (our Hunter kills an approaching Collector)
    s_k = [k for k in r["s_kills"] if k[1] in our_u and k[2] in opp_u and k[0] <= r["s_first_opp"] + 1]
    out["opp_walker_killed"] = len(s_k)
    if dfo < 0 and dfp >= 0:
        out["class"] = "ours_earlier"
    elif dfo == 0 and dfp > 0:
        out["class"] = "opp_later"
    elif dfo < 0 and dfp < 0:
        out["class"] = "both_earlier_ours_more"
    elif dfo > 0 and dfp > dfo:
        out["class"] = "ours_later_opp_much_later"
    else:
        out["class"] = "other"
    return out


def cmd_analyze(args):
    out = Path(args.out)
    P, S = load(out, "plain", args.opponent), load(out, "search", args.opponent)
    assert np.array_equal(P["seed"], S["seed"]) and np.array_equal(P["side"], S["side"])
    hunter = int(P["hunter"])
    branches = json.loads(str(S["branches"]))
    rows = []
    for g in range(len(P["seed"])):
        r = analyze_game(P, S, g, hunter)
        # sanity: identical moves before the first non-base branch
        nb = S["branch"][:, g] > 0
        t_nb = int(nb.argmax()) if nb.any() else S["moves"].shape[0]
        r["dev_before_branch"] = bool(r["dev_tick"] >= 0 and r["dev_tick"] < t_nb)
        r["first_branch_tick"] = t_nb if nb.any() else -1
        if r["s_race"] > r["p_race"]:
            r["mech"] = mechanism(r, P, S, g, hunter)
        rows.append(r)
    trans = Counter((r["p_race"], r["s_race"]) for r in rows)
    print(f"{args.opponent}: {len(rows)} matches; race (plain -> search) transitions:")
    for k, v in sorted(trans.items()):
        print(f"   {k[0]:+d} -> {k[1]:+d}: {v}")
    pr = Counter(r["p_race"] for r in rows)
    sr = Counter(r["s_race"] for r in rows)
    print(f"   plain w/t/l {pr[1]}/{pr[0]}/{pr[-1]}   search {sr[1]}/{sr[0]}/{sr[-1]}")
    print(f"   moves deviate before first non-base branch: {sum(r['dev_before_branch'] for r in rows)} (should be 0)")
    imp = [r for r in rows if "mech" in r]
    print(f"improved races: {len(imp)}; classes {Counter(r['mech']['class'] for r in imp)}")
    print(f"   d_first_ours {Counter(r['mech']['d_first_ours'] for r in imp).most_common(10)}")
    print(f"   d_first_opp  {Counter(min(r['mech']['d_first_opp'], 999) for r in imp).most_common(10)}")
    print(f"   same first-transforming unit: {sum(r['mech']['same_first_unit'] for r in imp)}/{len(imp)}")
    print(f"   branches executed pre-race: {Counter(branches[b] for r in imp for b in r['branches_pre_race'])}")
    print(f"   first non-base tick: {Counter(r['first_branch_tick'] for r in imp).most_common(8)}")
    (out / f"analysis_{args.opponent}.json").write_text(json.dumps(rows, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("play")
    q.add_argument("--opponent", default="strategic_v23")
    q.add_argument("--seeds", type=int, default=128)
    q.add_argument("--seed0", type=int, default=4_000_000)
    q.add_argument("--chunk", type=int, default=128)
    q.add_argument("--cap", type=int, default=300, help="decisions per match (25 = 1 s)")
    q.add_argument("--search-until", type=int, default=250)
    q.add_argument("--branches", default="alt*,samp:1.0x2")
    q.add_argument("--margin", type=float, default=3.0)
    q.add_argument("--threads", type=int, default=4)
    q.add_argument("--device", default="")
    q.add_argument("--force", action="store_true")
    q.add_argument("--out", default=str(REPO / "reports/race_analysis"))
    q = sub.add_parser("analyze")
    q.add_argument("--opponent", default="strategic_v23")
    q.add_argument("--out", default=str(REPO / "reports/race_analysis"))
    args = p.parse_args()
    {"play": cmd_play, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
