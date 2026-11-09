"""Second rollout actor in its own process (2026-11-09).

The trunk rollout is CPU-heavy numba code that holds the GIL, so a second actor thread would mostly serialise with the
first one; a separate process does not.  The child runs a collect-only ``PPOTrainer`` (the same, tested ``collect()``:
its own trunk games, opponents, heuristic workers, CUDA graphs) on ``actor2_device``.  Every iteration the parent sends
the behaviour policy + critic weights (the async actor's copies), the PFSP scores and the league snapshots; the child
plays one segment of its games and returns its batch, which the parent appends to its own.

Shared statistics: the PFSP score EMA starts from the same value on both sides; per opponent, the parent keeps its
update, takes the child's when only the child saw that opponent finish, and the mean of both updates otherwise (= one
EMA step towards the mean of the two batches' results).  ``fin`` (finished-game sums) is additive.  The per-batch
advantage normalisation / per-opponent centring stays per actor, and the league ramp windows (``fixed_win`` / snapshot
``win``) see the parent's half only.
"""
from __future__ import annotations

import atexit
import copy
import traceback

import numpy as np


def _sd_np(sd) -> dict:
    return {k: v.detach().cpu().numpy() for k, v in sd.items()}


def _child(cfg, conn) -> None:
    try:
        import torch

        from .ppo import PPOTrainer

        if torch.device(cfg.device).type == "cuda":
            torch.cuda.set_device(torch.device(cfg.device))
        tr = PPOTrainer(cfg)
        conn.send(("ready", None))
        while True:
            try:
                msg = conn.recv()
            except EOFError:   # parent gone
                return
            if msg[0] == "stop":
                return
            _, pol, cri, opp_score, snaps = msg
            for path, level, added, last in snaps[len(tr.snaps):]:   # same attach order as the parent -> same opponent indices
                tr._attach_snapshot(path, level, added, last)
            for sn, (_, level, _, _) in zip(tr.snaps, snaps):
                sn["level"] = level
            tr.net.load_state_dict({k: torch.from_numpy(v) for k, v in pol.items()})   # in place: CUDA graphs stay valid
            tr.critic.load_state_dict({k: torch.from_numpy(v) for k, v in cri.items()})
            assert len(opp_score) == len(tr.opps), (len(opp_score), len(tr.opps))
            tr.opp_score = np.asarray(opp_score, np.float64).copy()
            tr.opp_w = tr._pfsp_weights()
            fin0 = tr.fin.copy()
            batch, stats = tr.collect()
            conn.send(("ok", (batch, stats, tr.opp_score.copy(), tr.fin - fin0)))
    except BaseException:
        try:
            conn.send(("err", traceback.format_exc()))
        except Exception:
            pass


class Actor2:
    """Parent-side handle of the second actor process."""

    def __init__(self, cfg):
        import multiprocessing as mp

        c2 = copy.deepcopy(cfg)
        c2.device = cfg.actor2_device
        c2.actor_device = ""
        c2.critic_device = ""
        c2.actor2_device = ""
        c2.async_collect = "false"
        c2.league = False            # the parent decides snapshots; the child only attaches what it is sent
        c2.cuda_mem_fraction = 0.0
        c2.teach = ""
        if cfg.actor2_games > 0:
            c2.games = cfg.actor2_games
        c2.seed = cfg.seed + 100_003  # different trunk seeds / opponent draws / sampling
        ctx = mp.get_context("spawn")
        self.conn, child_conn = ctx.Pipe()
        # not daemonic: the child starts its own (daemonic) heuristic workers; it exits when the pipe closes
        self.proc = ctx.Process(target=_child, args=(c2, child_conn), name="ppo-actor2")
        self.proc.start()
        child_conn.close()
        self._ready = False
        self._inflight = False
        atexit.register(self.close)

    def _recv(self):
        kind, payload = self.conn.recv()
        if kind == "err":
            raise RuntimeError("actor2 process failed:\n" + payload)
        return kind, payload

    def request(self, act_net, act_critic, opp_score, snaps) -> None:
        if not self._ready:
            kind, _ = self._recv()
            assert kind == "ready", kind
            self._ready = True
        snaps_msg = [(s["path"], s["level"], s["added"], s["last_step"]) for s in snaps]
        self.conn.send(("collect", _sd_np(act_net.state_dict()), _sd_np(act_critic.state_dict()),
                        np.asarray(opp_score, np.float64).copy(), snaps_msg))
        self._inflight = True

    def result(self):
        kind, payload = self._recv()
        assert kind == "ok", kind
        self._inflight = False
        return payload

    def close(self) -> None:
        try:
            if self.proc.is_alive():
                if self._inflight:
                    self.conn.recv()
                self.conn.send(("stop",))
                self.proc.join(30)
        except Exception:
            pass
        if self.proc.is_alive():
            self.proc.terminate()


def merge_pfsp(s0: np.ndarray, s_parent: np.ndarray, s_child: np.ndarray) -> np.ndarray:
    """Combined PFSP EMA from two updates that both started at ``s0`` (see the module docstring)."""
    ch_p, ch_c = s_parent != s0, s_child != s0
    out = s_parent.copy()
    both = ch_p & ch_c
    out[both] = (s_parent[both] + s_child[both]) / 2.0
    only_c = ch_c & ~ch_p
    out[only_c] = s_child[only_c]
    return out


def merge_batches(b1: dict, b2: dict) -> dict:
    assert set(b1) == set(b2), (sorted(b1), sorted(b2))
    return {k: np.concatenate([b1[k], b2[k]]) for k in b1}
