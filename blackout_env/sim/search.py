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
K_BASE, K_ALT, K_ROT, K_SAMP, K_STOP, K_ALTM = 0, 1, 2, 3, 4, 5
KIND_NAMES = {K_BASE: "base", K_ALT: "alt", K_ROT: "rot", K_SAMP: "samp", K_STOP: "stop", K_ALTM: "pair"}
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
        if self.kind == K_ALT:
            return f"alt{self.unit}" + (f"r{round(self.param)}" if self.param else "")
        if self.kind == K_ALTM:
            return "pair" + "".join(str(u) for u in range(5) if (self.unit >> u) & 1)
        if self.kind == K_STOP:
            return f"{n}{self.unit}"
        if self.kind == K_ROT:
            return f"rot{self.unit}{'+' if self.param > 0 else '-'}{round(abs(math.degrees(self.param)))}"
        if self.kind == K_SAMP:
            return f"samp{self.param:g}s{self.seed}"
        return n


def parse_branches(spec: str) -> list[Branch]:
    """Comma list: ``base`` | ``alt*`` / ``alt3`` | ``rot*:90`` (+-90 deg, every unit) / ``rot2:45`` |
    ``samp:1.0x3`` (temperature 1, 3 seeds; ``samp:1.0x3@8``: seeds 8..10) | ``stop*`` |
    ``alt*:1`` (rank 1 = each unit's third choice: outside the top +- 1 and the second choice +- 1) |
    ``pair*`` (every two-unit joint second choice, 10) / ``pair03``.  The base branch is always first."""
    out = [Branch(K_BASE)]
    for tok in (t.strip() for t in spec.split(",")):
        if not tok or tok == "base":
            continue
        name, _, arg = tok.partition(":")
        if name.startswith("samp"):
            arg, _, s0 = (arg or "1.0x3").partition("@")
            temp, _, n = arg.partition("x")
            out += [Branch(K_SAMP, 0, float(temp), int(s0 or 0) + s) for s in range(int(n or 1))]
            continue
        if name.startswith("pair"):
            pairs = ([(a, b) for a in range(5) for b in range(a + 1, 5)] if name == "pair*"
                     else [(int(name[4]), int(name[5]))])
            out += [Branch(K_ALTM, (1 << a) | (1 << b)) for a, b in pairs]
            continue
        kind = {"alt": K_ALT, "rot": K_ROT, "stop": K_STOP}[name.rstrip("*0123456789")]
        units = range(5) if name.endswith("*") else [int(name[-1])]
        for u in units:
            if kind == K_ROT:
                deg = float(arg or 90)
                out += [Branch(K_ROT, u, math.radians(deg)), Branch(K_ROT, u, -math.radians(deg))]
            elif kind == K_ALT:
                out.append(Branch(K_ALT, u, float(arg or 0)))
            else:
                out.append(Branch(kind, u))
    return out


class ModNetPolicy(BatchPolicy):
    """A ``NetPolicy`` whose moves for one team of each game can carry a branch modification for
    the next ``left[b]`` decisions -- and, optionally, the other team of that game another one
    (``configure_opp``: the opponent-model variants of S4) for ``oleft[b]`` decisions.  Per-game
    state lives in arrays indexed by game; ``configure`` sets them (``reset`` keeps them, so
    ``BatchRollout.run`` does not wipe a configuration).  Each ``moves`` call is one decision for
    the games in ``idx`` (both teams in one call, as ``BatchRollout`` does with the same policy
    object on both sides and repeat 1)."""

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
        self.configure_opp(None)

    def configure_opp(self, kind=None, unit=None, param=None, seed=None, left=None):
        """The other team's modification in every game whose ``team`` is set (call after ``configure``)."""
        n = len(self.team)
        z = lambda dt, v=0: np.full(n, v, dt)  # noqa: E731
        self.okind = z(np.int64) if kind is None else np.asarray(kind, np.int64).copy()
        self.ounit = z(np.int64) if unit is None else np.asarray(unit, np.int64).copy()
        self.oparam = z(np.float64) if param is None else np.asarray(param, np.float64).copy()
        self.oseed = z(np.int64) if seed is None else np.asarray(seed, np.int64).copy()
        self.oleft = z(np.int64) if left is None else np.asarray(left, np.int64).copy()

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
        k = q.shape[-1]
        game_team = np.asarray(sel)[None, :].repeat(n, 0)          # [n, s]
        tm = self.team[idx][:, None]
        own = tm == game_team
        opp = (tm >= 0) & ~own
        act_own = own & (self.left[idx] > 0)[:, None]
        act_opp = opp & (self.oleft[idx] > 0)[:, None]
        kind = np.where(act_own, self.kind[idx][:, None], np.where(act_opp, self.okind[idx][:, None], K_BASE))
        pick = lambda a, b: np.where(own, a[idx][:, None], b[idx][:, None])  # noqa: E731
        unit, param, seed = pick(self.unit, self.ounit), pick(self.param, self.oparam), pick(self.seed, self.oseed)
        rows_n, rows_s = np.nonzero(kind == K_ALT)
        if len(rows_n):  # rank r = (r + 1)-th choice: mask the top bin +- 1, r more times
            u = unit[rows_n, rows_s]
            rank = np.rint(param[rows_n, rows_s]).astype(np.int64)
            for r in range(int(rank.max()) + 1):
                m = rank >= r
                rn, rs, ru = rows_n[m], rows_s[m], u[m]
                top = q[rn, rs, ru].argmax(-1)
                for d in (-1, 0, 1):
                    q[rn, rs, ru, (top + d) % k] = -np.inf
        rows_n, rows_s = np.nonzero(kind == K_ALTM)
        if len(rows_n):  # several units at once, each its second choice
            mask = unit[rows_n, rows_s]
            for u in range(5):
                m = ((mask >> u) & 1) == 1
                rn, rs = rows_n[m], rows_s[m]
                if len(rn):
                    top = q[rn, rs, u].argmax(-1)
                    for d in (-1, 0, 1):
                        q[rn, rs, u, (top + d) % k] = -np.inf
        rows_n, rows_s = np.nonzero(kind == K_SAMP)
        if len(rows_n):
            if self.noise is None or self.noise.shape[-1] != k:
                self.noise = _gumbel_table(k)
            b = idx[rows_n]
            temp = np.maximum(param[rows_n, rows_s], 1e-3)[:, None, None]
            nz = self.noise[seed[rows_n, rows_s] % NOISE_SEEDS, self.t[b] % NOISE_STEPS]
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
            ang = param[rows_n, rows_s]
            v = mv[rows_n, rows_s, u].astype(np.float64)
            c, s = np.cos(ang), np.sin(ang)
            rot = np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], -1)
            mv[rows_n, rows_s, u] = np.clip(rot, -1, 1).astype(F32)
        self.left[idx] = np.maximum(self.left[idx] - 1, 0)
        self.oleft[idx] = np.maximum(self.oleft[idx] - 1, 0)
        self.t[idx] += 1
        return {tm: mv[:, i] for i, tm in enumerate(sel)}


def branch_units(kind, unit) -> np.ndarray:
    """[m] branch kinds / units -> bool [m, 5]: the own units a branch modifies (base none, samp all)."""
    kind, unit = np.asarray(kind, np.int64), np.asarray(unit, np.int64)
    u = np.arange(5)[None, :]
    one = (kind[:, None] != K_ALTM) & (unit[:, None] == u)
    many = (kind[:, None] == K_ALTM) & (((unit[:, None] >> u) & 1) == 1)
    return np.where((kind == K_SAMP)[:, None], True, np.where((kind == K_BASE)[:, None], False, one | many))


def overlay_base(mod_mv, base_mv, kind, unit, overlay: bool) -> np.ndarray:
    """Own moves of games executing ``kind``/``unit``: the base policy's (``base_mv``) where the branch does not
    act -- every unit for the base branch, the untouched units of a unit branch (``overlay``) -- else the rollout
    net's modified ones (``mod_mv``).  ``overlay=False``: a non-base branch is the whole-team rollout net."""
    mask = branch_units(kind, unit)
    if not overlay:
        mask = np.repeat((np.asarray(kind) != K_BASE)[:, None], 5, 1)
    return np.where(mask[..., None], mod_mv, base_mv).astype(F32)


class ExternalBasePolicy(BatchPolicy):
    """Search rollouts with an external acting policy (docs §32): a ``ModNetPolicy`` whose searching team, while
    its branch is live (``left > 0``, the window), plays ``base`` (e.g. the P14b student, the V25 heuristic) for the
    base branch and for the units a unit branch leaves alone (``overlay``; ``samp`` = the whole team, it modifies
    every unit); the branch's targeted units play the rollout net's modified moves.  Past the window (and the
    other team always) = the rollout net, as without it.  ``reset`` resets ``base`` only (the branch configuration
    stays, as ``ModNetPolicy`` does)."""

    def __init__(self, mod: ModNetPolicy, base: BatchPolicy, overlay: bool = True):
        self.mod, self.base, self.overlay = mod, base, bool(overlay)
        self.accepts_packed = mod.accepts_packed and (getattr(base, "accepts_packed", False) or not base.needs_inputs)

    def reset(self, n: int) -> None:
        self.base.reset(n)

    def moves(self, bsim, idx, teams, inputs):
        mod = self.mod
        n = len(idx)
        live = (mod.team[idx] >= 0) & (mod.left[idx] > 0)          # before ``mod.moves`` counts the decision
        kind, unit, team = mod.kind[idx].copy(), mod.unit[idx].copy(), mod.team[idx].copy()
        out = mod.moves(bsim, idx, teams, inputs)
        for tm in teams:
            rows = np.nonzero(live & (team == tm))[0]
            if not len(rows):
                continue
            sub = tuple(x[:n][rows] for x in inputs)
            ext = self.base.moves(bsim, idx[rows], (tm,), sub)[tm]
            out[tm][rows] = overlay_base(out[tm][rows], ext, kind[rows], unit[rows], self.overlay)
        return out


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
    """Search over ``branches``: ``search(roots, teams, reseeds)`` scores every branch of every root in one batch.

    The S3 default is 1-ply: branch for ``window`` decisions, both teams the rollout net for
    ``leaf_decisions``, V_sym.  S4 options (docs §26):

    * ``opp_branches`` (base first): every own branch is also played against each opponent
      variant (the opponent's slot of ``ModNetPolicy``, for ``opp_left`` decisions, default the
      window); a branch's value = ``opp_agg`` ("min" = max-min, "mean" = expectation) over them.
    * ``children`` (base first): 2-ply -- after a branch's window, each of ``children`` is played
      for ``child_window`` decisions, then the leaf rollout; a branch's value = max over its children
      (our team will search again then).  ``expand`` k > 0: best-first -- only the base branch and the
      k best by 1-ply value get children (the rest score -inf); 0 = every branch.

    ``base_policy`` (docs §32): the window of the base branch -- and, with ``overlay``, the untouched units of a
    unit branch -- is played by this external policy instead of the rollout net (``ExternalBasePolicy``); the
    leaf rollouts stay the rollout net.  ``play_matches(base=...)`` then acts with it.  ``None`` = unchanged.
    """

    def __init__(self, policy: ModNetPolicy, leaf: ValueNetLeaf, branches: list[Branch], window: int = 25,
                 leaf_decisions: int = 125, margin: float = 0.0, packed: bool = False,
                 opp_branches: list[Branch] | None = None, opp_agg: str = "min", opp_left: int | None = None,
                 children: list[Branch] | None = None, child_window: int | None = None, expand: int = 0,
                 base_policy: BatchPolicy | None = None, overlay: bool = True):
        assert branches[0].kind == K_BASE
        assert opp_branches is None or opp_branches[0].kind == K_BASE
        assert children is None or children[0].kind == K_BASE
        assert opp_agg in ("min", "mean")
        assert not (opp_branches and children), "opponent variants and 2-ply are not combined"
        self.policy, self.leaf, self.branches = policy, leaf, branches
        self.window, self.leaf_decisions, self.margin = int(window), int(leaf_decisions), float(margin)
        self.opp_branches = opp_branches if opp_branches and len(opp_branches) > 1 else None
        self.opp_agg, self.opp_left = opp_agg, int(window if opp_left is None else opp_left)
        self.children = children if children and len(children) > 1 else None
        self.child_window = int(window if child_window is None else child_window)
        self.expand = int(expand)
        self.base_policy, self.overlay = base_policy, bool(overlay)
        rp = policy if base_policy is None else ExternalBasePolicy(policy, base_policy, overlay)
        self.roll = BatchRollout(rp, rp, repeat=1, packed=packed)

    def _configure(self, n_games, gteam, br: list[Branch], left):
        self.policy.configure(n_games, team=gteam, kind=[b.kind for b in br], unit=[b.unit for b in br],
                              param=[b.param for b in br], seed=[b.seed for b in br], left=np.full(n_games, left))

    def search(self, roots: list[SimState], teams, reseeds) -> SearchOut:
        if self.children is not None:
            return self._search2(roots, teams, reseeds)
        r, nb = len(roots), len(self.branches)
        no = len(self.opp_branches) if self.opp_branches else 1
        per = nb * no
        teams = np.asarray(teams, np.int64)
        sims = [s for s in roots for _ in range(per)]
        gteam = np.repeat(teams, per)
        br = [b for b in self.branches for _ in range(no)] * r      # game (root, branch, opponent variant)
        self._configure(len(sims), gteam, br, self.window)
        if self.opp_branches:
            ob = self.opp_branches * (r * nb)
            self.policy.configure_opp([b.kind for b in ob], [b.unit for b in ob], [b.param for b in ob],
                                      [b.seed + 32 * (b.kind == K_SAMP) for b in ob],  # draws other than ours
                                      np.full(len(sims), self.opp_left))
        c0 = time.perf_counter()
        res = self.roll.run(sims, self.window + self.leaf_decisions, reseeds=np.repeat(np.asarray(reseeds), per))
        c1 = time.perf_counter()
        raw = self.leaf(res.bsim, gteam).reshape(r, nb, no)
        vals = raw[..., 0] if no == 1 else (raw.min(-1) if self.opp_agg == "min" else raw.mean(-1))
        c2 = time.perf_counter()
        return SearchOut(choose(vals, self.margin), vals,
                         {"rollout_s": c1 - c0, "leaf_s": c2 - c1, "games": len(sims),
                          "game_decisions": self.roll.timing["game_decisions"], "raw": raw if no > 1 else None})

    def _search2(self, roots, teams, reseeds) -> SearchOut:
        r, nb, nc = len(roots), len(self.branches), len(self.children)
        teams = np.asarray(teams, np.int64)
        gteam = np.repeat(teams, nb)
        c0 = time.perf_counter()
        gd = 0
        self._configure(r * nb, gteam, self.branches * r, self.window)
        res = self.roll.run([s for s in roots for _ in range(nb)], self.window, reseeds=np.repeat(np.asarray(reseeds), nb))
        gd += self.roll.timing["game_decisions"]
        parents = res.bsim
        v1 = None
        if self.expand > 0:  # 1-ply values of every branch (continue each parent: the rollout net both teams)
            cont = take_games(parents, np.arange(parents.n))
            self.policy.configure(cont.n)
            res1 = self.roll.run(None, self.leaf_decisions, bsim=cont)
            gd += self.roll.timing["game_decisions"]
            v1 = self.leaf(res1.bsim, gteam).reshape(r, nb)
            order = np.argsort(-v1[:, 1:], axis=1)[:, :self.expand] + 1
            expanded = np.zeros((r, nb), bool)
            expanded[:, 0] = True
            expanded[np.arange(r)[:, None], order] = True
        else:
            expanded = np.ones((r, nb), bool)
        pr = np.nonzero(expanded.reshape(-1))[0]                    # expanded parent games
        kids = take_games(parents, np.repeat(pr, nc))
        kteam = np.repeat(gteam[pr], nc)
        self._configure(kids.n, kteam, self.children * len(pr), self.child_window)
        res2 = self.roll.run(None, self.child_window + self.leaf_decisions, bsim=kids)
        gd += self.roll.timing["game_decisions"]
        c1 = time.perf_counter()
        kv = self.leaf(res2.bsim, kteam).reshape(len(pr), nc)
        vals = np.full(r * nb, -np.inf)
        vals[pr] = kv.max(1)
        vals = vals.reshape(r, nb)
        c2 = time.perf_counter()
        return SearchOut(choose(vals, self.margin), vals,
                         {"rollout_s": c1 - c0, "leaf_s": c2 - c1, "games": int(r * nb + kids.n),
                          "game_decisions": gd, "v1": v1})

    def config(self) -> dict:
        """What a search result depends on besides the nets (``search_server`` handshake)."""
        return search_config(self.branches, self.window, self.leaf_decisions, self.opp_branches, self.opp_agg,
                             self.opp_left, self.children, self.child_window, self.expand)

    def games_per_root(self) -> int:
        """Rollout games per root of the largest stage (CUDA-graph bucket sizing)."""
        nb = len(self.branches)
        if self.opp_branches:
            return nb * len(self.opp_branches)
        if self.children:
            return max(nb, (self.expand + 1 if self.expand else nb) * len(self.children))
        return nb


DEEP_KEYS = ("leaf_decisions", "opp_branches", "opp_agg", "opp_left", "children", "child_window", "expand")


def searcher_kwargs(deep: dict | None) -> dict:
    """S4 options (branch lists as ``parse_branches`` specs, ``None``/empty = off) -> ``Searcher`` kwargs."""
    kw = {}
    for k, v in (deep or {}).items():
        if k not in DEEP_KEYS:
            raise ValueError(f"unknown search option {k!r}")
        if v in (None, ""):
            continue
        kw[k] = parse_branches(v) if k in ("opp_branches", "children") else v
    return kw


def search_config(branches, window=25, leaf_decisions=125, opp_branches=None, opp_agg="min", opp_left=None,
                  children=None, child_window=None, expand=0) -> dict:
    """``Searcher.config()`` from its arguments (branch lists or specs for ``parse_branches``)."""
    names = lambda b: [x.name for x in (parse_branches(b) if isinstance(b, str) else b)]  # noqa: E731
    out = {"branches": names(branches), "window": int(window), "leaf_decisions": int(leaf_decisions)}
    ob = names(opp_branches) if opp_branches else []
    if len(ob) > 1:
        out |= {"opp_branches": ob, "opp_agg": opp_agg, "opp_left": int(window if opp_left is None else opp_left)}
    ch = names(children) if children else []
    if len(ch) > 1:
        out |= {"children": ch, "child_window": int(window if child_window is None else child_window), "expand": int(expand)}
    return out


def take_games(bsim: BatchSim, idx) -> BatchSim:
    """A new ``BatchSim`` holding copies of games ``idx`` of ``bsim`` (repeats allowed): the
    mid-rollout states of a search tree's parents, without a round trip through ``SimState``."""
    from .batch import _SIM_ARRAYS

    idx = np.asarray(idx, np.int64)
    out = object.__new__(BatchSim)
    out.n, out.order = len(idx), bsim.order
    for name in _SIM_ARRAYS:
        setattr(out, name, np.ascontiguousarray(getattr(bsim, name)[idx]))
    out.Mb = tuple(np.ascontiguousarray(m[idx]) for m in bsim.Mb)
    out.bg = np.ascontiguousarray(bsim.bg[idx])
    out.layouts = [bsim.layouts[i] for i in idx]
    out.all = np.arange(out.n, dtype=np.int64)
    return out


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
    search_from: int = 0       # search only at decisions search_from <= t < search_until (None = no end)
    search_until: int | None = None
    trace_s: tuple = ()        # also return our lead after these match seconds (the final lead if over)


def play_matches(seeds, sides, our: ModNetPolicy, opponent: BatchPolicy, searcher: Searcher | None,
                 cfg: MatchConfig, log=None, base: BatchPolicy | None = None, overlay: bool = True) -> dict:
    """Lockstep matches from ``SimState.from_seed`` (Unity RNG): team ``sides[b]`` is ours (the
    rollout net, searching every ``cfg.every`` decisions when ``searcher`` is given), the other team
    ``opponent``.  ``base`` (docs §32): our team acts with this policy instead of the rollout net
    except where a non-base branch is executing (its targeted units, or with ``overlay=False`` the
    whole team, play the rollout net's modified moves).  Returns per-match results (+ recorded rows)."""
    n = len(seeds)
    sides = np.asarray(sides, np.int64)
    bs = BatchSim([SimState.from_seed(int(s)) for s in seeds])
    our.configure(n)
    opponent.reset(n)
    if base is not None:
        base.reset(n)
    trace_t = [int(round(x * 25)) for x in cfg.trace_s]
    lead_at = np.full((n, len(trace_t)), np.iinfo(np.int64).min, np.int64)
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
    search_gd = 0
    cap = cfg.cap_s * 25
    t0 = time.perf_counter()
    for t in range(cap):
        live = np.nonzero(~bs.over)[0]
        if not len(live):
            break
        if (searcher is not None and t % cfg.every == 0 and t >= cfg.search_from
                and (cfg.search_until is None or t < cfg.search_until)):
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
            search_gd += int(out.timing.get("game_decisions", 0))
            t_search += time.perf_counter() - c0
        c0 = time.perf_counter()
        bs.render(live, *bufs)
        sub = tuple(x[: len(live)] for x in bufs)
        ours = sides[live]
        executing = np.where(our.left[live] > 0, cur_branch[live], 0).astype(np.int16)
        pre_left, pre_kind, pre_unit = our.left[live].copy(), our.kind[live].copy(), our.unit[live].copy()
        if base is not None and opponent is not our and not (pre_left > 0).any():
            mo = {tm: np.zeros((len(live), 5, 2), F32) for tm in (0, 1)}  # the base acts alone: skip the net
            our.t[live] += 1
        else:
            mo = our.moves(bs, live, (0, 1), sub)  # both views in one forward (the r5 opponent's too)
        if base is not None:
            for team in (0, 1):
                rows = np.nonzero(ours == team)[0]
                if not len(rows):
                    continue
                s_in = sub if len(rows) == len(live) else tuple(x[rows] for x in sub)
                ext = base.moves(bs, live[rows], (team,), s_in)[team]
                kd = np.where(pre_left[rows] > 0, pre_kind[rows], K_BASE)
                mo[team][rows] = overlay_base(mo[team][rows], ext, kd, pre_unit[rows], overlay)
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
        for i, at in enumerate(trace_t):
            if at == t + 1:
                ld = bs.scores[live, 0] - bs.scores[live, 1]
                lead_at[live, i] = np.where(sides[live] == 0, ld, -ld)
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
           "gain_sum": gain_sum, "wall_s": time.perf_counter() - t0, "search_s": t_search, "play_s": t_play,
           "search_game_decisions": search_gd}
    if trace_t:
        fin = (ours_score - opp_score)[:, None]
        out["lead_at"] = np.where(lead_at == np.iinfo(np.int64).min, fin, lead_at)
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
