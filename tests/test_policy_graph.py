"""ForkPolicyGraph (the exported stateless half of the official submission) == MyPolicy on fork obs."""

from pathlib import Path

import numpy as np
import pytest
import torch

CKPT = Path(__file__).resolve().parents[1] / "checkpoints/distill/frozen/p24c_v321.pt"


@pytest.mark.skipif(not CKPT.is_file(), reason="P24c checkpoint not fetched (scripts/fetch_checkpoints.sh)")
def test_graph_matches_policy_both_teams():
    from blackout_env.official.policy_graph import ForkPolicyGraph, reference_moves
    from blackout_env.sim.state import SimState
    from blackout_env.train.distill.inference import load_checkpoint_net, make_student

    net = load_checkpoint_net(str(CKPT))
    policy, graph = make_student(net, "iqn"), ForkPolicyGraph(net)
    rng = np.random.default_rng(0)
    sim = SimState.from_seed(5)
    for step in range(60):
        if step % 6 == 0:
            for team in (0, 1):
                rs, rm = sim.raw_state(), sim.raw_map()
                with torch.no_grad():
                    got = graph(torch.from_numpy(rs), torch.from_numpy(rm.astype(np.int64)), torch.tensor(team))[0].numpy()
                np.testing.assert_allclose(got, reference_moves(policy, rs, rm, team), atol=1e-5)
        sim.step(rng.uniform(-1, 1, (10, 2)).astype(np.float32))
