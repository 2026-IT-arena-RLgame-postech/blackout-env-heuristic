import json

import numpy as np

from blackout_env.heuristics import HeuristicPolicyMixture
from blackout_env.train.offline_dataset import load_dataset_into, save_buffer
from blackout_env.train.policy_strength import POLICY_IDS, bc_weight_table, policy_index
from blackout_env.train.replay_buffer import SequentialReplayBuffer


def test_bc_weights_mean_one_under_mixture_and_ordered_by_elo():
    weights = HeuristicPolicyMixture(seed=0).weights
    table = bc_weight_table(weights)
    mean = sum(w * table[policy_index(p)] for p, w in weights.items()) / sum(weights.values())
    assert abs(mean - 1.0) < 1e-4
    assert table[policy_index("strategic_v19")] > table[policy_index("strategic_v17")] > table[policy_index("strategic_v4")] > table[policy_index("strategic_v1")]
    assert table[-1] == 1.0 and policy_index("not_a_policy") == -1


def _buffer(n):
    return SequentialReplayBuffer(capacity=n, graphic_shape=(2, 2, 1), team_state_size=4, agent_state_size=3)


def test_policy_and_demo_survive_save_and_load(tmp_path):
    src = _buffer(8)
    for i in range(6):
        src.push(np.zeros((2, 2, 1)), np.zeros(4), np.zeros((10, 3)), np.zeros(10, dtype=np.int64), 0.0, i == 5,
                 policy=i % 3 - 1)
    save_buffer(src, tmp_path / "buffer_a.npz")
    (tmp_path / "collection.json").write_text(json.dumps({"demo": False}))
    dst = _buffer(8)
    assert load_dataset_into(dst, tmp_path / "buffer_a.npz") == 6
    assert list(dst.policy[:6]) == [-1, 0, 1, -1, 0, 1]
    assert not dst.demo[:6].any()
    assert len(POLICY_IDS) == 20


def test_gaussian_heading_noise_stays_centred_on_heuristic():
    from blackout_env.env.constants import team_a_agents, team_b_agents
    from blackout_env.train.qmix_trainer import QMIXConfig, QMIXTrainer

    trainer = QMIXTrainer(env=None, config=QMIXConfig(buffer_capacity=4, tb_log_dir=None, heuristic_noise_mode="gaussian",
                                                      heuristic_noise_sigma_deg=20.0))
    east = {a: np.array([1.0, 0.0]) for a in team_a_agents() + team_b_agents()}
    trainer.heuristic_a.act = lambda obs: {a: east[a] for a in obs}
    trainer.heuristic_b.act = lambda obs: {a: east[a] for a in obs}
    obs = {a: {} for a in team_a_agents() + team_b_agents()}
    np.random.seed(0)
    picks = np.concatenate([trainer.select_actions_heuristic(obs)[1] for _ in range(2000)])
    share = np.bincount(picks, minlength=8) / len(picks)
    assert 0.70 < share[0] < 0.80            # P(|N(0, 20 deg)| < 22.5 deg) = 0.74
    assert 0.09 < share[1] < 0.17 and 0.09 < share[7] < 0.17
    assert share[2:7].sum() < 0.01


def test_sampled_bc_weight_is_demo_times_policy_strength():
    from blackout_env.train.qmix_trainer import QMIXConfig, QMIXTrainer

    trainer = QMIXTrainer(env=None, config=QMIXConfig(buffer_capacity=64, tb_log_dir=None, bc_policy_weighting=True))
    buf = trainer.buffer_a
    graphic = np.zeros(buf.graphic.shape[1:], dtype=np.float32)
    v19, v1 = policy_index("strategic_v19"), policy_index("strategic_v1")
    for i in range(64):
        buf.push(graphic, np.zeros(buf.team_state.shape[1:]), np.zeros(buf.agent_states.shape[1:]), np.zeros(10, dtype=np.int64),
                 0.0, i % 16 == 15, demo=i % 4 != 3, policy=v19 if i % 2 else v1)
    batch = trainer._sample_batch(buf, 32, n_step=3, gamma=0.99, beta=0.4)
    expected = np.where(batch["demo"], trainer._bc_policy_weight[np.where(batch["indices"] % 2 == 1, v19, v1)], 0.0)
    assert np.allclose(batch["bc_weight"], expected)
    assert trainer._bc_policy_weight[v19] > 1.5 > 0.3 > trainer._bc_policy_weight[v1]
