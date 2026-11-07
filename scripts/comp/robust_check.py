"""Robustness of a stateful team-mode ONNX submission to what the official server does (onnx_policy.TeamPolicy):

* absent units: their rows are zero-padded and their state rows are NOT updated (kept stale);
* skipped steps (timeout / error): no call at all -> no state update for anyone, the team stands still.

A reference game is played in our 50 Hz sim (ONNX = team ``--side``, B6 = the other team), recording the official
observations of every step.  The same observation stream is then replayed (open loop) with random absences / skips,
and the present units' actions are compared with the reference ones.  Mismatch = the tracker state got corrupted.

    PYTHONPATH=.:$HOME/project/26rl/ort_pkgs .venv/bin/python scripts/comp/robust_check.py SUB_DIR [--side 0|1]
"""
import argparse
import os
import sys

os.environ.setdefault("BLACKOUT_TPD", "1")
sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from blackout_env.official.render import official_preprocessed  # noqa: E402
from blackout_env.rl.grpo import _heuristic_aliases  # noqa: E402
from blackout_env.sim.batch import BatchSim, HeuristicPolicy  # noqa: E402
from blackout_env.sim.state import SimState  # noqa: E402


class Server:
    """onnx_policy.TeamPolicy semantics: per-slot state, zero-padded absent rows, absent states kept."""

    def __init__(self, path):
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.S = self.sess.get_inputs()[2].shape[1]
        self.reset()

    def reset(self):
        self.state = np.zeros((5, self.S), np.float32)

    def act(self, vec, gra, slots):
        v = np.zeros((5, 96), np.float32)
        g = np.zeros((5, 11, 96, 96), np.float32)
        v[slots], g[slots] = vec[slots], gra[slots]
        a, s = self.sess.run(None, {"vector": v, "graphic": g, "state_in": self.state})
        self.state[slots] = s[slots]
        return a


def obs(sim, side):
    o = [official_preprocessed(sim, side * 5 + i) for i in range(5)]
    return (np.stack([x["vector"] for x in o]),
            np.stack([x["graphic"].transpose(2, 0, 1) for x in o]).astype(np.float32))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sub")
    p.add_argument("--side", type=int, default=0)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--p-absent", type=float, default=0.1, help="per unit-step absence probability (bursts)")
    p.add_argument("--p-skip", type=float, default=0.01)
    a = p.parse_args()
    srv = Server(os.path.join(a.sub, "model.onnx"))
    opp = HeuristicPolicy(_heuristic_aliases()["B6"], a.seed)
    bs = BatchSim([SimState.from_seed(a.seed)])
    bs.ticks_per_decision = 1
    opp.reset(1)
    idx = np.array([0])
    all5 = np.arange(5)
    stream, ref = [], []
    moves = np.zeros((1, 10, 2), np.float32)
    for t in range(a.steps):
        if bs.over[0]:
            break
        v, g = obs(bs.state(0), a.side)
        act = srv.act(v, g, all5)
        stream.append((v, g)); ref.append(act)
        mine = slice(5 * a.side, 5 * a.side + 5)
        theirs = slice(5 - 5 * a.side, 10 - 5 * a.side)
        moves[0, mine] = act
        moves[0, theirs] = opp.moves(bs, idx, (1 - a.side,), None)[1 - a.side]
        bs.step(np.clip(moves, -1, 1), idx)
    n = len(stream)
    print(f"reference: {n} steps, final score {bs.scores[0].tolist()} (side {a.side})")
    rng = np.random.default_rng(a.seed)

    def replay(absent_p, skip_p, label):
        srv.reset()
        out = [0, 0, 0, 0, 0]    # compared unit-steps, mismatches, skipped, all-present unit-steps, their mismatches
        gone = np.zeros(5, int)  # remaining absence length per slot (bursts of 25-150 steps, like a respawn wait)
        for t, ((v, g), r) in enumerate(zip(stream, ref)):
            start = (gone == 0) & (rng.random(5) < absent_p / 50)
            gone[start] = rng.integers(25, 150, start.sum())
            present = np.nonzero(gone == 0)[0]
            gone = np.maximum(gone - 1, 0)
            if rng.random() < skip_p:
                out[2] += 1
                continue
            if not len(present):
                continue
            act = srv.act(v, g, present)
            diff = np.abs(act[present] - r[present]).max(axis=1) > 1e-3
            out[0] += len(present); out[1] += int(diff.sum())
            if len(present) == 5:
                out[3] += 5; out[4] += int(diff.sum())
        print(f"{label:26s} mismatch {out[1] / max(out[0], 1):6.2%} of {out[0]} unit-steps | all-present steps "
              f"{out[4] / max(out[3], 1):6.2%} of {out[3]} | skipped {out[2]}")

    replay(0.0, 0.0, "identical replay")
    replay(a.p_absent, 0.0, "absent units (zero rows)")
    replay(0.0, a.p_skip, "skipped steps (timeouts)")
    replay(a.p_absent, a.p_skip, "both")


if __name__ == "__main__":
    main()
