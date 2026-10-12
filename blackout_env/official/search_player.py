"""The R6 search on top of the tracker: P24c acts, the r24cd256 rollout net searches (docs §61 ``o4f_r24c``).

Exactly ``play_matches``' own-team logic at batch 1, with the tracker's reconstruction as the
root: from decision ``search_from`` (5 s) every ``every`` decisions, ``Searcher.search`` scores
the branches ``alt*,samp:1.0x2`` (opponent model: rollout-net samples x3 + its arg max, mean;
leaf V s2v_f3xd2 iter2; margin 3) and the chosen one runs for ``window`` decisions -- its
targeted units play the rollout net's modified moves, every other unit the base policy (overlay).

A decision is two official ticks (the fork's decision period), so the simulator the search rolls
out is the official game itself; only the opponent's future moves are modelled.

    player = SearchPlayer(base_ckpt, rollout_ckpt, value_ckpt)    # paths or file-like objects
    player.new_match()
    moves = player.act(sim, team, decision, base_moves)            # base_moves: P24c's [5, 2]
"""

from __future__ import annotations

import time

import numpy as np

from ..sim.batch import N_AGENT_COLS, N_CH, BatchSim, NetPolicy
from ..sim.mapdata import H, W
from ..sim.rollout_net import RolloutNet
from ..sim.search import (K_BASE, ModNetPolicy, Searcher, ValueNetLeaf, overlay_base, parse_branches,
                          searcher_kwargs)

F32 = np.float32
BRANCHES = "alt*,samp:1.0x2"
DEEP = {"opp_branches": "samp:1.0x3", "opp_left": 150, "opp_agg": "mean"}
SEARCH_FROM, EVERY, WINDOW, MARGIN = 125, 25, 25, 3.0


def _load(src):
    import torch

    if hasattr(src, "seek"):
        src.seek(0)
    return torch.load(src, map_location="cpu", weights_only=False)


class SearchPlayer:
    def __init__(self, base_ckpt, rollout_ckpt, value_ckpt, device: str = "cpu", search_from: int = SEARCH_FROM,
                 every: int = EVERY, window: int = WINDOW, margin: float = MARGIN):
        from ..model.my_model import MyModel
        from ..value.model import load_value_net

        sd = _load(base_ckpt)["policy_state"]
        base_net = MyModel.from_state_dict(sd)
        base_net.load_state_dict(sd)
        self.base = NetPolicy(base_net.eval(), device=device)           # the base branch inside the rollouts
        ck = _load(rollout_ckpt)
        rnet = RolloutNet(**ck["config"])
        rnet.load_state_dict(ck["state"])
        roll = NetPolicy(rnet.eval(), device=device)
        if hasattr(value_ckpt, "seek"):
            value_ckpt.seek(0)
        leaf = ValueNetLeaf(load_value_net(value_ckpt, device=device), device)
        self.searcher = Searcher(ModNetPolicy(roll), leaf, parse_branches(BRANCHES), window=window, margin=margin,
                                 base_policy=self.base, overlay=True, **searcher_kwargs(DEEP))
        self.mod = {0: ModNetPolicy(roll), 1: ModNetPolicy(roll)}     # our acting state, per side
        self.search_from, self.every, self.window = search_from, every, window
        self.bufs = (np.zeros((1, 2, N_CH, H, W), F32), np.zeros((1, 2, 4), F32), np.zeros((1, 2, 10, N_AGENT_COLS), F32))
        self.last = None   # (decision, reseed, SearchOut) of the last search
        self.stats = {"searches": 0, "search_s": 0.0, "worst_s": 0.0, "chosen": np.zeros(len(self.searcher.branches), np.int64)}
        self.new_match()

    def new_match(self) -> None:
        for m in self.mod.values():
            m.configure(1)
        self.stats["chosen"][:] = 0

    def act(self, sim, team: int, decision: int, base_moves: np.ndarray, reseed: int = 0) -> np.ndarray:
        """Our team's moves for this decision (``sim``: the tracker's state at it, not modified)."""
        mod, idx = self.mod[team], np.zeros(1, np.int64)
        if decision >= self.search_from and decision % self.every == 0:
            c0 = time.perf_counter()
            out = self.searcher.search([sim.copy()], [team], [int(reseed) % (2 ** 31)])
            k = int(out.choice[0])
            self.last = (decision, int(reseed) % (2 ** 31), out)
            mod.set_game(0, team, self.searcher.branches[k], self.window)
            dt = time.perf_counter() - c0
            self.stats["searches"] += 1
            self.stats["search_s"] += dt
            self.stats["worst_s"] = max(self.stats["worst_s"], dt)
            self.stats["chosen"][k] += 1
        kind = int(mod.kind[0]) if mod.left[0] > 0 else K_BASE
        if kind == K_BASE:  # count the decision as ModNetPolicy.moves would
            mod.left[0] = max(int(mod.left[0]) - 1, 0)
            mod.oleft[0] = max(int(mod.oleft[0]) - 1, 0)
            mod.t[0] += 1
            return np.asarray(base_moves, F32)
        bs = BatchSim([sim.copy()])
        bs.render(idx, *self.bufs)
        mv = mod.moves(bs, idx, (team,), self.bufs)[team]
        return overlay_base(mv, np.asarray(base_moves, F32)[None], [kind], [int(mod.unit[0])], True)[0]
