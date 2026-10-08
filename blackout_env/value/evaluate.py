"""Batch evaluation of the S1 value net on live observations (simulator or BlackOutEnv), with the
input frame the net was trained in and an optional mirror-symmetry average.

Training rows (train/distill/collect.py ``_labelled_row``) are one learner view mapped by
``team_frame.canonical_obs`` -- a team B view is anti-transposed and its unit rows rolled so the
own team is rows 0-4 -- with the graphic in the shard's uint8 encoding (x15).  ``encode_view``
does exactly that to any env-format view, so V sees simulator states the way it saw shards.

Mirror symmetry.  The same state seen from the other team (its view through the same
canonicalisation) is a second, equally valid input: the game is symmetric under the map's y = x
reflection with the teams swapped, so a perfect V would satisfy V(s) = -V(mirror(s)) for every
lead quantity and P_win(s) = P_loss(mirror(s)).  The net only approximately does; averaging the
two readings,

    lead quantities  q_sym(s) = (q(s) - q(mirror(s))) / 2
    result probs     p_sym(s) = (p_{loss,draw,win}(s) + p_{win,draw,loss}(mirror(s))) / 2

is antisymmetric by construction and cancels any side bias and part of the per-view noise.
``symmetrize`` is that average on two decoded predictions; ``ValueEvaluator.evaluate_obs``
applies it to whole observation dicts.
"""

from __future__ import annotations

import numpy as np
import torch

from blackout_env.env.team_frame import canonical_obs
from blackout_env.train.distill.dataset import encode_graphic
from blackout_env.value.model import decode_outputs, load_value_net

TEAM_VIEW = ("unit_0", "unit_5")  # any unit of a team carries its team's (shared) view


def encode_view(view: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One env-format view -> (graphic uint8 [24, 24, 13], team_state f32 [4], agent_states f32 [10, 12])
    in the canonical (training) frame."""
    canonical = canonical_obs(view)
    return (encode_graphic(np.asarray(canonical["graphic"], np.float32)),
            np.asarray(canonical["team_state"], np.float32), np.asarray(canonical["agent_states"], np.float32))


def team_view(obs: dict, team: int) -> dict:
    """Team ``team``'s (0 = A, 1 = B) view from a full observation dict (sim.obs())."""
    name = TEAM_VIEW[team]
    if name in obs:
        return obs[name]
    lo = 5 * team
    return next(v for k, v in obs.items() if lo <= int(k.split("_")[1]) < lo + 5)


def encode_batch(views: list[dict]) -> dict[str, np.ndarray]:
    enc = [encode_view(v) for v in views]
    return {"graphic": np.stack([e[0] for e in enc]), "team_state": np.stack([e[1] for e in enc]),
            "agent_states": np.stack([e[2] for e in enc])}


def symmetrize(own: dict, mirror: dict) -> dict:
    """Antisymmetric average of the decoded predictions of a state (``own``) and of the same state
    from the other team (``mirror``); both as ``decode_outputs`` returns them (+ optional ``lead``)."""
    out = {"delta": (own["delta"] - mirror["delta"]) / 2.0, "final": (own["final"] - mirror["final"]) / 2.0,
           "p": (own["p"] + mirror["p"][:, ::-1]) / 2.0}
    out["p_win"] = out["p"][:, 2]
    if "lead" in own:
        out["lead"] = (own["lead"] - mirror["lead"]) / 2.0  # = own lead exactly
    return out


class ValueEvaluator:
    """Batched V on encoded rows or env views.  Predictions are in score/100 units (as the net's
    targets): ``delta`` [N, H] lead change over HORIZONS_S, ``final`` [N] final lead, ``p`` [N, 3]
    (loss, draw, win), ``p_win`` [N], ``lead`` [N] the current lead."""

    def __init__(self, path_or_net, device: str = "cpu", batch: int = 256):
        self.device = torch.device(device)
        self.net = (load_value_net(path_or_net, device=self.device) if isinstance(path_or_net, (str, bytes))
                    or hasattr(path_or_net, "__fspath__") else path_or_net.to(self.device).eval())
        self.batch = int(batch)

    @torch.no_grad()
    def evaluate_encoded(self, graphic: np.ndarray, team_state: np.ndarray, agent_states: np.ndarray) -> dict:
        parts = []
        for a in range(0, len(graphic), self.batch):
            b = a + self.batch
            g = torch.from_numpy(np.ascontiguousarray(graphic[a:b])).to(self.device)
            ts = torch.from_numpy(np.ascontiguousarray(team_state[a:b], dtype=np.float32)).to(self.device)
            ag = torch.from_numpy(np.ascontiguousarray(agent_states[a:b], dtype=np.float32)).to(self.device)
            out = decode_outputs(self.net.forward_u8(g, ts, ag).float(), ts[:, 0] - ts[:, 1])
            out["lead"] = (ts[:, 0] - ts[:, 1]).cpu().numpy()
            parts.append(out)
        if not parts:
            raise ValueError("nothing to evaluate")
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    def evaluate_views(self, views: list[dict]) -> dict:
        enc = encode_batch(views)
        return self.evaluate_encoded(enc["graphic"], enc["team_state"], enc["agent_states"])

    def evaluate_obs(self, obs_list: list[dict], teams, symmetric: bool = False) -> dict:
        """V of each full observation dict from team ``teams[i]``'s side (0 = A, 1 = B);
        ``symmetric`` averages with the other team's reading (``symmetrize``)."""
        teams = np.broadcast_to(np.asarray(teams, dtype=np.int64), (len(obs_list),))
        own = self.evaluate_views([team_view(o, int(t)) for o, t in zip(obs_list, teams)])
        if not symmetric:
            return own
        mirror = self.evaluate_views([team_view(o, 1 - int(t)) for o, t in zip(obs_list, teams)])
        return symmetrize(own, mirror)
