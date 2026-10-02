"""S1 value evaluation on live observations (blackout_env/value/evaluate.py): the training frame
and the mirror-symmetry average."""

import numpy as np
import pytest
import torch

from blackout_env.env.team_frame import mirror_agent_states, mirror_graphic
from blackout_env.sim import SimState
from blackout_env.train.distill.dataset import encode_graphic
from blackout_env.value.evaluate import ValueEvaluator, encode_view, symmetrize, team_view
from blackout_env.value.model import ValueNet


def _played_obs(seed=950123, steps=120):
    sim = SimState.from_seed(seed)
    rng = np.random.default_rng(seed)
    out = []
    for step in range(steps):
        sim.step(rng.uniform(-1, 1, (10, 2)).astype(np.float32))
        if step % 40 == 39:
            out.append(sim.obs())
    return out


@pytest.fixture(scope="module")
def obs_list():
    return _played_obs()


def test_encode_view_matches_the_distill_shard_frame(obs_list):
    obs = obs_list[-1]
    a, b = team_view(obs, 0), team_view(obs, 1)
    ga, ta, sa = encode_view(a)
    gb, tb, sb = encode_view(b)
    # team A is already canonical; team B is anti-transposed with its own units in rows 0-4
    assert np.array_equal(ga, encode_graphic(a["graphic"])) and np.array_equal(sa, a["agent_states"])
    assert np.array_equal(gb, encode_graphic(mirror_graphic(b["graphic"])))
    assert np.array_equal(sb, mirror_agent_states(b["agent_states"]))
    assert (sa[:5, 2] > 0).all() and (sa[5:, 2] < 0).all() and (sb[:5, 2] > 0).all() and (sb[5:, 2] < 0).all()
    assert tb[0] == ta[1] and tb[1] == ta[0]
    assert ga.dtype == np.uint8 and ga.shape == (24, 24, 13)


def test_symmetrize_is_antisymmetric():
    rng = np.random.default_rng(0)

    def pred(n=5):
        p = rng.dirichlet(np.ones(3), n)
        return {"delta": rng.normal(size=(n, 4)), "final": rng.normal(size=n), "p": p, "p_win": p[:, 2],
                "lead": rng.normal(size=n)}

    x, y = pred(), pred()
    xy, yx = symmetrize(x, y), symmetrize(y, x)
    assert np.allclose(xy["final"], -yx["final"]) and np.allclose(xy["delta"], -yx["delta"])
    assert np.allclose(xy["p"], yx["p"][:, ::-1]) and np.allclose(xy["p"].sum(1), 1.0)


def test_symmetric_value_is_antisymmetric_between_the_two_teams(obs_list):
    torch.manual_seed(0)
    ev = ValueEvaluator(ValueNet(hidden_size=64), batch=2)
    a = ev.evaluate_obs(obs_list, 0, symmetric=True)
    b = ev.evaluate_obs(obs_list, 1, symmetric=True)
    assert np.allclose(a["final"], -b["final"], atol=1e-6) and np.allclose(a["delta"], -b["delta"], atol=1e-6)
    assert np.allclose(a["p_win"], b["p"][:, 0], atol=1e-6)
    assert np.allclose(a["lead"], -b["lead"]) and np.allclose(a["lead"], ev.evaluate_obs(obs_list, 0)["lead"])
    # the plain readings are not antisymmetric (a random net has no reason to be)
    pa, pb = ev.evaluate_obs(obs_list, 0), ev.evaluate_obs(obs_list, 1)
    assert not np.allclose(pa["final"], -pb["final"], atol=1e-4)


def test_branch_metrics_on_a_synthetic_root():
    from examples import value_branch_eval as m

    B, w = len(m.BRANCHES), m.WINDOWS[-1]
    finals = np.arange(B, dtype=float) * 2.0  # branch b is worth 2b points; V23 (b = 0) is the worst
    finals[0] = 0.0
    branches, preds = [], {}
    for b in range(B):
        row = {"terminal": False, "reached": w, "lead_end": 0, "pos_hash": b, "capped": 0,
               "final_r": [finals[b]] * 4, "lead_h_r": [finals[b]] * 4, "win_r": [1.0] * 4}
        branches.append({"name": m.BRANCHES[b], "windows": {str(w): row}, "race": [None, None]})
        v = -finals[b]  # V ranks exactly backwards
        p = {"lead": 0.0, "v_final": v, "m_final": -v, "v_d": [0, 0, 0, v], "m_d": [0, 0, 0, -v],
             "v_p": [0.2, 0.0, 0.8], "m_p": [0.8, 0.0, 0.2], "h_final": 0.0, "h_d20": 0.0, "h_pwin": 0.5}
        for j in range(max(m.AVG_K)):
            preds[f"{b}|{w}|t{j}"] = p
        for k in m.LEAF_ROLL:
            preds[f"{b}|{w}|roll{k}"] = {**p, "v_final": finals[b], "m_final": -finals[b]}
    root = {"branches": branches, "preds": preds, "t_s": 1.0}
    met = m.root_metrics(root, w)
    assert met["oracle"] == finals.max() and met["no_search"] == 0.0 and met["oracle_cv"] == finals.max()
    assert met["pick"]["v_k1"] == finals.min() and met["pick"]["v_roll125"] == finals.max()
    assert met["pick"]["lead"] == pytest.approx(finals.mean())  # all tied: uniform over branches
    s = m.summarize([met], [root])
    assert s["concordance"]["final"]["v_k1"][">=0"][0] == 0.0
    assert s["concordance"]["final"]["v_roll125"][">=0"][0] == 1.0
    assert s["regret"]["v_roll125"] == 0.0 and s["gain_vs_no_search"]["v_roll125"][0] == finals.max()
