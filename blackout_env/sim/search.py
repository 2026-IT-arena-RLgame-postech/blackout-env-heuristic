"""S3 self-play search teacher: 1-ply search over r5 branches with "5 s r5 x r5 rollout + V_sym" leaves.
(docs/policy_v2_design_ko.md §20 "S3 자기 대전 탐색 교사".)

At a search point (one root state per game, our team T) every candidate ``Branch`` is played for
``window`` decisions -- our team: the rollout net with that branch's modification, the opponent:
the plain rollout net (the self-play model) -- then both teams play the plain rollout net for
``leaf`` decisions, and the leaf is scored by the mirror-symmetric final-lead head of the value
net (exact final lead when the match ended).  Every root's B branches share one reseed (common
random numbers) and all roots x branches run as one ``BatchRollout`` batch.  The branch with the
highest leaf value wins (the base branch unless another beats it by ``margin`` points) and our
team executes it, closed loop, until the next search.

Branches (all expressible by the student's head -- K = 32 bins + offset, no standing move -- so a
search decision is a distillable label):

* ``base``         the rollout net's arg max (no change);
* ``alt``  unit u  unit u's best bin outside the net's top bin +- 1 (its second-choice direction);
* ``rot``  unit u  unit u's move rotated by ``param`` radians (world frame);
* ``samp`` seed s  every own unit's bin sampled from softmax(logits / ``param``) (Gumbel noise
                   from a fixed table, so the search and the executed branch see the same draws);
* ``stop`` unit u  unit u stands still (not distillable by a no-STAND head: off by default).

    from blackout_env.sim.search import ModNetPolicy, Searcher, parse_branches
    pol = ModNetPolicy(NetPolicy(RolloutNet.from_checkpoint(R5), device="mps"))
    s = Searcher(pol, ValueNetLeaf("checkpoints/value/frozen/s2v_iter5.pt", "mps"), parse_branches("base,alt*"))
    out = s.search(roots, teams, reseeds)       # out.choice [R], out.values [R, B]
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import core
from .batch import N_AGENT_COLS, N_CH, BatchPolicy, BatchRollout, BatchSim, NetPolicy, decode_moves
from .mapdata import HUNTER, H, W
from .state import SimState

F32 = np.float32
K_BASE, K_ALT, K_ROT, K_SAMP, K_STOP = 0, 1, 2, 3, 4
KIND_NAMES = {K_BASE: "base", K_ALT: "alt", K_ROT: "rot", K_SAMP: "samp", K_STOP: "stop"}
NOISE_SEEDS, NOISE_STEPS = 64, 256


def default_device() -> str:
    """cuda > mps > cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _gumbel_table(k: int = 32) -> np.ndarray:
    rng = np.random.default_rng(20261003)
    u = rng.random((NOISE_SEEDS, NOISE_STEPS, 5, k))
    return (-np.log(-np.log(np.clip(u, 1e-12, 1 - 1e-12)))).astype(F32)


@dataclass(frozen=True)
class Branch:
    kind: int
    unit: int = 0
    param: float = 0.0
    seed: int = 0

    @property
    def name(self) -> str:
        n = KIND_NAMES[self.kind]
        if self.kind in (K_ALT, K_STOP):
            return f"{n}{self.unit}"
        if self.kind == K_ROT:
            return f"rot{self.unit}{'+' if self.param > 0 else '-'}{round(abs(math.degrees(self.param)))}"
        if self.kind == K_SAMP:
            return f"samp{self.param:g}s{self.seed}"
        return n


def parse_branches(spec: str) -> list[Branch]:
    """Comma list: ``base`` | ``alt*`` / ``alt3`` | ``rot*:90`` (+-90 deg, every unit) / ``rot2:45`` |
    ``samp:1.0x3`` (temperature 1, 3 seeds) | ``stop*``.  The base branch is always first."""
    out = [Branch(K_BASE)]
    for tok in (t.strip() for t in spec.split(",")):
        if not tok or tok == "base":
            continue
        name, _, arg = tok.partition(":")
        if name.startswith("samp"):
            temp, _, n = (arg or "1.0x3").partition("x")
            out += [Branch(K_SAMP, 0, float(temp), s) for s in range(int(n or 1))]
            continue
        kind = {"alt": K_ALT, "rot": K_ROT, "stop": K_STOP}[name.rstrip("*0123456789")]
        units = range(5) if name.endswith("*") else [int(name[-1])]
        for u in units:
            if kind == K_ROT:
                deg = float(arg or 90)
                out += [Branch(K_ROT, u, math.radians(deg)), Branch(K_ROT, u, -math.radians(deg))]
            else:
                out.append(Branch(kind, u))
    return out


class ModNetPolicy(BatchPolicy):
    """A ``NetPolicy`` whose moves for one team of each game can carry a branch modification for
    the next ``left[b]`` decisions.  Per-game state lives in arrays indexed by game; ``configure``
    sets them (``reset`` keeps them, so ``BatchRollout.run`` does not wipe a configuration).
    Each ``moves`` call is one decision for the games in ``idx`` (both teams in one call, as
    ``BatchRollout`` does with the same policy object on both sides and repeat 1)."""

    def __init__(self, net: NetPolicy):
        self.net = net
        self.accepts_packed = getattr(net, "accepts_packed", False)
        self.noise = None
        self.configure(0)

    def configure(self, n, team=None, kind=None, unit=None, param=None, seed=None, left=None):
        z = lambda dt, v=0: np.full(n, v, dt)  # noqa: E731
        self.team = z(np.int64, -1) if team is None else np.asarray(team, np.int64).copy()
        self.kind = z(np.int64) if kind is None else np.asarray(kind, np.int64).copy()
        self.unit = z(np.int64) if unit is None else np.asarray(unit, np.int64).copy()
        self.param = z(np.float64) if param is None else np.asarray(param, np.float64).copy()
        self.seed = z(np.int64) if seed is None else np.asarray(seed, np.int64).copy()
        self.left = z(np.int64) if left is None else np.asarray(left, np.int64).copy()
        self.t = z(np.int64)

    def set_game(self, b, team, br: Branch, left: int):
        self.team[b], self.kind[b], self.unit[b], self.param[b], self.seed[b] = team, br.kind, br.unit, br.param, br.seed
        self.left[b], self.t[b] = left, 0

    def moves(self, bsim, idx, teams, inputs):
        graphic, team_in, agents = inputs
        n, sel = len(idx), list(teams)
        cut = slice(None) if sel == [0, 1] else sel
        g = graphic[:n][:, cut].reshape(n * len(sel), *graphic.shape[2:])
        t = team_in[:n][:, cut].reshape(n * len(sel), 4)
        a = agents[:n][:, cut].reshape(n * len(sel), 10, N_AGENT_COLS)
        q, off = self.net.forward(g, t, a)
        q = q.reshape(n, len(sel), 5, -1)
        game_team = np.asarray(sel)[None, :].repeat(n, 0)          # [n, s]
        active = (self.left[idx] > 0)[:, None] & (self.team[idx][:, None] == game_team)  # [n, s]
        kind = np.where(active, self.kind[idx][:, None], K_BASE)
        unit = self.unit[idx][:, None].repeat(len(sel), 1)
        rows_n, rows_s = np.nonzero(kind == K_ALT)
        if len(rows_n):
            u = unit[rows_n, rows_s]
            k = q.shape[-1]
            top = q[rows_n, rows_s, u].argmax(-1)
            for d in (-1, 0, 1):
                q[rows_n, rows_s, u, (top + d) % k] = -np.inf
        rows_n, rows_s = np.nonzero(kind == K_SAMP)
        if len(rows_n):
            if self.noise is None or self.noise.shape[-1] != q.shape[-1]:
                self.noise = _gumbel_table(q.shape[-1])
            b = idx[rows_n]
            temp = np.maximum(self.param[b], 1e-3)[:, None, None]
            nz = self.noise[self.seed[b] % NOISE_SEEDS, self.t[b] % NOISE_STEPS]
            q[rows_n, rows_s] = q[rows_n, rows_s] / temp + nz
        q = q.reshape(n * len(sel), 5, -1)
        team_b = np.tile(np.asarray(sel) == 1, n)
        mv = decode_moves(q, off, team_b).reshape(n, len(sel), 5, 2)
        rows_n, rows_s = np.nonzero(kind == K_STOP)
        if len(rows_n):
            mv[rows_n, rows_s, unit[rows_n, rows_s]] = 0.0
        rows_n, rows_s = np.nonzero(kind == K_ROT)
        if len(rows_n):
            u = unit[rows_n, rows_s]
            ang = self.param[idx[rows_n]]
            v = mv[rows_n, rows_s, u].astype(np.float64)
            c, s = np.cos(ang), np.sin(ang)
            rot = np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], -1)
            mv[rows_n, rows_s, u] = np.clip(rot, -1, 1).astype(F32)
        self.left[idx] = np.maximum(self.left[idx] - 1, 0)
        self.t[idx] += 1
        return {tm: mv[:, i] for i, tm in enumerate(sel)}


class ValueNetLeaf:
    """Leaf scorer: the value net's final-lead head, mirror-symmetric ((V(own) - V(other)) / 2), points,
    on the batch renderer's canonical inputs (= value/evaluate.encode_view's frame)."""

    def __init__(self, path_or_net, device: str = "cpu", batch: int = 2048, fp16: bool = False):
        import torch

        from blackout_env.value.model import load_value_net

        self.torch, self.device = torch, torch.device(device)
        self.net = (load_value_net(path_or_net, device=self.device) if isinstance(path_or_net, (str, bytes))
                    or hasattr(path_or_net, "__fspath__") else path_or_net.to(self.device).eval())
        self.batch, self.fp16 = int(batch), fp16

    def final(self, graphic, team, agents) -> np.ndarray:
        """[m, 13, H, W], [m, 4], [m, 10, 12] -> predicted final lead of the viewing team, points."""
        torch = self.torch
        from blackout_env.value.model import decode_outputs

        out = []
        for s in range(0, len(graphic), self.batch):
            g = torch.from_numpy(np.ascontiguousarray(graphic[s:s + self.batch])).to(self.device)
            t = torch.from_numpy(np.ascontiguousarray(team[s:s + self.batch])).to(self.device)
            a = torch.from_numpy(np.ascontiguousarray(agents[s:s + self.batch])).to(self.device)
            with torch.inference_mode():
                if self.fp16:
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                        raw = self.net(g, t, a)
                else:
                    raw = self.net(g, t, a)
                out.append(decode_outputs(raw.float(), t[:, 0] - t[:, 1])["final"])
        return np.concatenate(out) * 100.0

    def __call__(self, bsim: BatchSim, teams: np.ndarray) -> np.ndarray:
        """Symmetric value of every game of ``bsim`` for ``teams[b]`` (exact final lead if over)."""
        n = bsim.n
        lead = bsim.scores[:, 0] - bsim.scores[:, 1]
        val = np.where(teams == 0, lead, -lead).astype(np.float64)
        live = np.nonzero(~bsim.over)[0]
        if len(live):
            gr = np.zeros((len(live), 2, N_CH, H, W), F32)
            tm = np.zeros((len(live), 2, 4), F32)
            ag = np.zeros((len(live), 2, 10, N_AGENT_COLS), F32)
            bsim.render(live, gr, tm, ag)
            f = self.final(gr.reshape(-1, N_CH, H, W), tm.reshape(-1, 4), ag.reshape(-1, 10, N_AGENT_COLS)).reshape(-1, 2)
            own = np.where(teams[live] == 0, f[:, 0], f[:, 1])
            mir = np.where(teams[live] == 0, f[:, 1], f[:, 0])
            val[live] = (own - mir) / 2.0
        assert len(val) == n
        return val


@dataclass
class SearchOut:
    choice: np.ndarray        # [R] chosen branch index
    values: np.ndarray        # [R, B] leaf values (points, our team)
    timing: dict = field(default_factory=dict)


class Searcher:
    """1-ply search: ``search(roots, teams, reseeds)`` scores every branch of every root in one batch."""

    def __init__(self, policy: ModNetPolicy, leaf: ValueNetLeaf, branches: list[Branch], window: int = 25,
                 leaf_decisions: int = 125, margin: float = 0.0, packed: bool = False):
        assert branches[0].kind == K_BASE
        self.policy, self.leaf, self.branches = policy, leaf, branches
        self.window, self.leaf_decisions, self.margin = int(window), int(leaf_decisions), float(margin)
        self.roll = BatchRollout(policy, policy, repeat=1, packed=packed)

    def search(self, roots: list[SimState], teams, reseeds) -> SearchOut:
        r, nb = len(roots), len(self.branches)
        teams = np.asarray(teams, np.int64)
        sims = [s for s in roots for _ in range(nb)]
        gteam = np.repeat(teams, nb)
        br = self.branches * r
        self.policy.configure(len(sims), team=gteam, kind=[b.kind for b in br], unit=[b.unit for b in br],
                              param=[b.param for b in br], seed=[b.seed for b in br],
                              left=np.full(len(sims), self.window))
        c0 = time.perf_counter()
        res = self.roll.run(sims, self.window + self.leaf_decisions, reseeds=np.repeat(np.asarray(reseeds), nb))
        c1 = time.perf_counter()
        vals = self.leaf(res.bsim, gteam).reshape(r, nb)
        c2 = time.perf_counter()
        return SearchOut(choose(vals, self.margin), vals, {"rollout_s": c1 - c0, "leaf_s": c2 - c1, "games": len(sims)})

    def config(self) -> dict:
        """What a search result depends on besides the nets (``search_server`` handshake)."""
        return {"branches": [b.name for b in self.branches], "window": self.window,
                "leaf_decisions": self.leaf_decisions}


def choose(values: np.ndarray, margin) -> np.ndarray:
    """[R, B] leaf values -> [R] chosen branch: the best one if it beats the base branch (0) by more
    than ``margin`` points (scalar or [R]), else the base branch."""
    values = np.asarray(values)
    best = values.argmax(1)
    gain = values[np.arange(len(values)), best] - values[:, 0]
    return np.where(gain > np.asarray(margin, np.float64), best, 0)


class QSearcher:
    """The learned stand-in for ``Searcher`` (docs §22 C): a RolloutNet with a Q head predicts, per own
    unit, the leaf-value gain of the ``alt u`` branch over ``base``; the best unit is switched to its
    alternative direction for the window when its predicted gain exceeds ``margin``.  Same interface
    (``search(roots, teams, reseeds)`` -> SearchOut over ``branches`` = base + alt0..4), so
    ``play_matches`` executes the choice exactly like a search decision.  No rollouts."""

    def __init__(self, net, device: str, margin: float = 3.0, detach: bool = True, random_rate: float = 0.0, seed: int = 0):
        """``random_rate`` > 0: a control -- ignore the head, switch a uniformly random unit with this
        probability per search point (same deviation rate, no information)."""
        import torch

        self.random_rate = float(random_rate)
        self.rng = np.random.default_rng(seed)
        self.torch, self.device = torch, torch.device(device)
        self.net = net.to(self.device).eval()
        self.branches = parse_branches("alt*")
        self.margin = float(margin)

    def predict(self, graphic, team, agents) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            g = torch.from_numpy(np.ascontiguousarray(graphic)).to(self.device)
            t = torch.from_numpy(np.ascontiguousarray(team)).to(self.device)
            a = torch.from_numpy(np.ascontiguousarray(agents)).to(self.device)
            return self.net.q_alt(g, t, a)[2].float().cpu().numpy()

    def search(self, roots: list[SimState], teams, reseeds=None) -> SearchOut:
        r = len(roots)
        teams = np.asarray(teams, np.int64)
        if self.random_rate > 0:
            go = self.rng.random(r) < self.random_rate
            choice = np.where(go, 1 + self.rng.integers(0, 5, r), 0)
            return SearchOut(choice, np.zeros((r, 6)), {"games": r})
        bs = BatchSim(roots)
        gr = np.zeros((r, 2, N_CH, H, W), F32)
        tm = np.zeros((r, 2, 4), F32)
        ag = np.zeros((r, 2, 10, N_AGENT_COLS), F32)
        bs.render(np.arange(r), gr, tm, ag)
        j = np.arange(r)
        adv = self.predict(gr[j, teams], tm[j, teams], ag[j, teams])          # [r, 5]
        vals = np.concatenate([np.zeros((r, 1)), adv], 1)
        best = vals.argmax(1)
        choice = np.where(vals[j, best] > self.margin, best, 0)
        return SearchOut(choice, vals, {"games": r})


# ---------------------------------------------------------------------------------- matches


@dataclass
class MatchConfig:
    every: int = 25            # decisions between searches (25 = 1 s)
    window: int = 25           # decisions a chosen branch is executed / simulated
    cap_s: int = 150
    record: bool = False       # store the search team's (input, executed move) rows
    record_roots: bool = False  # store every search root: input, all B branch leaf values, r5 top / alt bins


def play_matches(seeds, sides, our: ModNetPolicy, opponent: BatchPolicy, searcher: Searcher | None,
                 cfg: MatchConfig, log=None) -> dict:
    """Lockstep matches from ``SimState.from_seed`` (Unity RNG): team ``sides[b]`` is ours (the
    rollout net, searching every ``cfg.every`` decisions when ``searcher`` is given), the other team
    ``opponent``.  Returns per-match results (+ recorded rows)."""
    n = len(seeds)
    sides = np.asarray(sides, np.int64)
    bs = BatchSim([SimState.from_seed(int(s)) for s in seeds])
    our.configure(n)
    opponent.reset(n)
    bufs = (np.zeros((n, 2, N_CH, H, W), F32), np.zeros((n, 2, 4), F32), np.zeros((n, 2, 10, N_AGENT_COLS), F32))
    moves = np.zeros((n, 10, 2), F32)
    first = np.full((n, 2), -1, np.int64)
    nb = len(searcher.branches) if searcher else 1
    chosen = np.zeros((n, nb), np.int64)
    gain_sum = np.zeros(n)
    searches = np.zeros(n, np.int64)
    rows = {k: [] for k in ("game", "tick", "map", "team_state", "agent_states", "moves", "branch")} if cfg.record else None
    cur_branch = np.zeros(n, np.int64)
    roots_rec = {k: [] for k in ("game", "tick", "map", "team_state", "agent_states", "values", "choice",
                                 "top_bin", "alt_bin")} if cfg.record_roots else None
    searched_now = None
    t_search = t_play = 0.0
    cap = cfg.cap_s * 25
    t0 = time.perf_counter()
    for t in range(cap):
        live = np.nonzero(~bs.over)[0]
        if not len(live):
            break
        if searcher is not None and t % cfg.every == 0:
            c0 = time.perf_counter()
            roots = [bs.state(int(b)) for b in live]
            reseeds = [(int(seeds[b]) * 7919 + t * 104729 + 12345) % (2 ** 31) for b in live]
            out = searcher.search(roots, sides[live], reseeds)
            for j, b in enumerate(live):
                k = int(out.choice[j])
                our.set_game(b, sides[b], searcher.branches[k], cfg.window)
                chosen[b, k] += 1
                gain_sum[b] += out.values[j, k] - out.values[j, 0]
                cur_branch[b] = k
            searches[live] += 1
            searched_now = (live.copy(), out)
            t_search += time.perf_counter() - c0
        c0 = time.perf_counter()
        bs.render(live, *bufs)
        sub = tuple(x[: len(live)] for x in bufs)
        ours = sides[live]
        executing = np.where(our.left[live] > 0, cur_branch[live], 0).astype(np.int16)
        mo = our.moves(bs, live, (0, 1), sub)  # both views in one forward (the r5 opponent's too)
        if opponent is our:
            oo = mo
        else:
            oo = {tm: np.zeros((len(live), 5, 2), F32) for tm in (0, 1)}
            for team in (0, 1):
                sel = ours != team  # the opponent plays ``team`` in these games
                if sel.any():
                    s_in = sub if sel.all() else tuple(x[sel] for x in sub)
                    oo[team][sel] = opponent.moves(bs, live[sel], (team,), s_in)[team]
        for team in (0, 1):
            mine = ours == team
            moves[live, 5 * team:5 * team + 5] = np.where(mine[:, None, None], mo[team], oo[team])
        if cfg.record_roots and searched_now is not None:  # inputs rendered at the root tick (search used copies)
            sl, so = searched_now
            assert np.array_equal(sl, live)
            j = np.arange(len(live))
            g_r, t_r, a_r = bufs[0][j, ours], bufs[1][j, ours], bufs[2][j, ours]
            q, _ = our.net.forward(np.ascontiguousarray(g_r), np.ascontiguousarray(t_r), np.ascontiguousarray(a_r))
            top = q.argmax(-1)
            qm = q.copy()
            k = q.shape[-1]
            for dlt in (-1, 0, 1):
                np.put_along_axis(qm, ((top + dlt) % k)[..., None], -np.inf, -1)
            roots_rec["game"].append(live.copy())
            roots_rec["tick"].append(np.full(len(live), t, np.int32))
            roots_rec["map"].append(_pack(g_r))
            roots_rec["team_state"].append(t_r.copy())
            roots_rec["agent_states"].append(a_r.copy())
            roots_rec["values"].append(so.values.astype(np.float32))
            roots_rec["choice"].append(so.choice.astype(np.int16))
            roots_rec["top_bin"].append(top.astype(np.int8))
            roots_rec["alt_bin"].append(qm.argmax(-1).astype(np.int8))
        searched_now = None
        if cfg.record:
            j = np.arange(len(live))
            rows["game"].append(live.copy())
            rows["tick"].append(np.full(len(live), t, np.int32))
            rows["map"].append(_pack(bufs[0][j, ours]))
            rows["team_state"].append(bufs[1][j, ours].copy())
            rows["agent_states"].append(bufs[2][j, ours].copy())
            rows["moves"].append(np.where((ours == 0)[:, None, None], moves[live, :5], moves[live, 5:]).copy())
            rows["branch"].append(executing)
        bs.step(moves, live)
        t_play += time.perf_counter() - c0
        for team in (0, 1):
            hit = (first[live, team] < 0) & (bs.ui[live, 5 * team:5 * team + 5, core.U_CLS] == HUNTER).any(axis=1)
            first[live[hit], team] = t + 1
        if log and t % 250 == 0:
            log(f"  t={t // 25}s live {len(live)} search {t_search:.0f}s play {t_play:.0f}s")
    sc = bs.scores
    ours_score = np.where(sides == 0, sc[:, 0], sc[:, 1])
    opp_score = np.where(sides == 0, sc[:, 1], sc[:, 0])
    f_ours = np.where(sides == 0, first[:, 0], first[:, 1])
    f_opp = np.where(sides == 0, first[:, 1], first[:, 0])
    out = {"seed": np.asarray(seeds), "side": sides, "ours": ours_score, "opp": opp_score,
           "margin": ours_score - opp_score, "over": bs.over.copy(), "ticks": bs.ticks,
           "first_ours": f_ours, "first_opp": f_opp, "searches": searches, "chosen": chosen,
           "gain_sum": gain_sum, "wall_s": time.perf_counter() - t0, "search_s": t_search, "play_s": t_play}
    if cfg.record:
        out["rows"] = {k: np.concatenate(v) for k, v in rows.items()}
    if cfg.record_roots:
        out["roots"] = {k: np.concatenate(v) for k, v in roots_rec.items()} if roots_rec["game"] else {}
    return out


def _pack(graphic: np.ndarray) -> np.ndarray:
    """float [n, 13, H, W] -> uint16 [n, H, W] (examples/rollout_net_train.pack_graphic's layout)."""
    base = graphic[:, :8].argmax(1).astype(np.uint16)
    bat = np.rint(graphic[:, 8] * 15).astype(np.uint16)
    items = graphic[:, 9:13]
    item = np.where(items.max(1) > 0, items.argmax(1) + 1, 0).astype(np.uint16)
    return base | (bat << 3) | (item << 7)


def race(first_ours: np.ndarray, first_opp: np.ndarray) -> np.ndarray:
    """+1 our first Hunter strictly earlier, 0 same tick (or neither), -1 later."""
    a = np.where(first_ours < 0, np.iinfo(np.int64).max, first_ours)
    b = np.where(first_opp < 0, np.iinfo(np.int64).max, first_opp)
    return np.sign(b - a)
