"""eval_batched (one rollout for all opponents x seeds x sides) == eval_matches, game for game (CPU, fp32)."""
import os

import numpy as np
import pytest

CK = "checkpoints/distill/frozen/p22_v31.pt"
pytestmark = pytest.mark.skipif(not os.path.exists(CK), reason="checkpoint missing")


def _old(net, spec, seeds, horizon):
    from blackout_env.rl.grpo import make_opponent_policy
    from blackout_env.sim.batch import BatchRollout, NetPolicy, SimState

    pol = NetPolicy(net, device="cpu")
    opp = make_opponent_policy(spec, "cpu")
    roots = [SimState.from_seed(s) for s in seeds]
    out = []
    for side in (0, 1):
        pa, pb = (pol, opp) if side == 0 else (opp, pol)
        out.append(BatchRollout(pa, pb, packed=True).run(roots, horizon).lead(side).astype(float))
    return out


@pytest.mark.parametrize("opps", [[f"net:{CK}", "V32.1"], [f"net:{CK}", "random"]])
def test_eval_batched_equals_eval_matches(opps):
    import torch

    from blackout_env.rl.evalfast import eval_batched
    from blackout_env.train.distill.inference import load_net

    net = load_net(torch.load(CK, map_location="cpu", weights_only=True)["policy_state"])
    seeds, horizon = [11, 12, 13], 400
    new = eval_batched(net, opps, seeds, "cpu", horizon=horizon, graphs=False, fp16=False)
    for spec in opps:
        m0, m1 = _old(net, spec, seeds, horizon)
        assert np.array_equal(new[spec]["margin_side0"], m0), (spec, new[spec]["margin_side0"], m0)
        assert np.array_equal(new[spec]["margin_side1"], m1), (spec, new[spec]["margin_side1"], m1)
        m = np.concatenate([m0, m1])
        assert abs(new[spec]["margin"] - m.mean()) < 1e-9
        assert abs(new[spec]["win"] - ((m > 0) + 0.5 * (m == 0)).mean()) < 1e-9
