import numpy as np
import pytest
import torch

from blackout_env.env.constants import all_agents
from blackout_env.env.team_frame import DIR_MIRROR
from blackout_env.heuristics.teach import TeachLabel
from blackout_env.model.my_model import MyModel
from blackout_env.model.my_policy import DIRECTION_VECTORS, MyPolicy
from blackout_env.train.distill.collect import MatchTask, _labelled_row, play_labelled_match
from blackout_env.train.distill.metrics import STAND, LabelAccuracy, summarize_matches
from blackout_env.train.distill.dataset import (
    KIND_INDEX, DistillDataset, decode_graphic, encode_graphic, save_shard, shard_meta, unpack_labels,
)
from blackout_env.train.distill.loss import set_valued_cross_entropy
from blackout_env.train.distill.run import class_weights


def _view(team_b: bool):
    """A team's observation of one small map (tests/test_v17_teach.py layout)."""
    graphic = np.zeros((24, 24, 13), dtype=np.float32)
    graphic[..., 0] = 1
    graphic[2:4, 5:7, 3] = 1
    graphic[11:13, 11:13, 2] = 1
    graphic[1:4, 1:4, 7 if team_b else 6] = 1
    graphic[20:23, 20:23, 6 if team_b else 7] = 1
    graphic[0, 0, 5 if team_b else 4] = 1
    graphic[23, 23, 4 if team_b else 5] = 1
    graphic[18, 6, 8] = 8 / 15
    graphic[6, 18, 8] = 9 / 15
    graphic[12, 3, 8] = 10 / 15
    states = np.zeros((10, 12), dtype=np.float32)
    for i in range(10):
        states[i, :2] = (-0.5 + 0.1 * i, 0.4 - 0.1 * i) if i < 5 else (0.6, -0.6 + 0.05 * i)
        states[i, 2] = (1 if i < 5 else -1) * (-1 if team_b else 1)
        states[i, 3] = 1
        states[i, 9] = 1
    return {"graphic": graphic, "team_state": np.array([0, 0, 1, 1], np.float32), "agent_states": states}


class _FakeEnv:
    """Fixed observations for ``length`` ticks, then team A wins 0.3 to 0.1."""

    def __init__(self, length=5):
        self.length, self.agents = length, []

    def reset(self, seed=None):
        self.t, self.agents = 0, list(all_agents())
        return self._obs(), {}

    def _obs(self):
        return {name: _view(team_b=i >= 5) for i, name in enumerate(all_agents())}

    def step(self, actions):
        assert all(np.shape(a) == (2,) for a in actions.values())
        self.t += 1
        if self.t < self.length:
            return self._obs(), {}, {a: False for a in self.agents}, {}, {}
        info = {"winner": 0, "score_0": 0.3, "score_1": 0.1}
        self.agents = []
        return {}, {}, {a: True for a in all_agents()}, {}, {a: info for a in all_agents()}


def test_graphic_encoding_is_lossless_and_refuses_other_values():
    graphic = _view(False)["graphic"]
    assert np.array_equal(decode_graphic(encode_graphic(graphic)), graphic)
    graphic[0, 0, 8] = 0.5
    with pytest.raises(ValueError):
        encode_graphic(graphic)


def _rows(n, start):
    return {
        "graphic": np.zeros((n, 2, 2, 1), np.uint8), "team_state": np.zeros((n, 4), np.float32),
        "agent_states": np.zeros((n, 10, 12), np.float32),
        "labels": (start + np.arange(n))[:, None].repeat(5, 1).astype(np.uint8), "kinds": np.zeros((n, 5), np.uint8),
    }


def test_dataset_ring_keeps_the_newest_rows_and_loads_each_shard_once(tmp_path):
    save_shard(tmp_path / "a.npz", _rows(3, 0))
    dataset = DistillDataset(tmp_path, capacity=4)
    assert dataset.refresh() == 3 and dataset.refresh() == 0
    save_shard(tmp_path / "b.npz", _rows(3, 10))
    assert dataset.refresh() == 3 and dataset.size == 4
    kept = set(dataset.sample(64, np.random.default_rng(0))["labels"][:, 0])
    assert kept == {2, 10, 11, 12}


def test_set_valued_ce_is_the_mass_on_the_set_and_ignores_unlabelled_rows():
    logits = torch.randn(2, 8, requires_grad=True)
    masks = torch.zeros(2, 8, dtype=torch.bool)
    masks[0, [1, 2]] = True  # row 1 unlabelled
    loss, stats = set_valued_cross_entropy(logits, masks, z_loss=0.0)
    expected = -torch.log(torch.softmax(logits[0], -1)[[1, 2]].sum())
    assert torch.allclose(loss, expected, atol=1e-6)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and torch.all(logits.grad[1] == 0)
    assert stats["accuracy"] == float(logits[0].argmax() in (1, 2))


def test_team_b_rows_are_canonical_own_units_first_with_mirrored_labels():
    obs = {name: _view(team_b=True) for name in all_agents()[5:]}
    labels = {}
    for k, name in enumerate(obs):
        mask = np.zeros(8, bool)
        mask[k] = True
        labels[name] = TeachLabel(mask, True, "battery", (0, 0))
    moves = {name: DIRECTION_VECTORS[k] for k, name in enumerate(obs)}
    row = _labelled_row(obs, labels, moves, {}, {name: True for name in obs})
    assert (row["agent_states"][:5, 2] > 0).all()  # own team in rows 0-4
    for k in range(5):
        assert np.flatnonzero(unpack_labels(row["labels"])[k]).tolist() == [int(DIR_MIRROR[k])]
        assert row["actions"][k] == DIR_MIRROR[k]  # executed moves mirror like labels
    assert (row["kinds"] == 1).all() and (row["student"] == STAND).all() and row["teacher_drove"].all()


@pytest.mark.parametrize("swap", [False, True])
def test_labelled_match_records_the_learner_team(swap):
    task = MatchTask(seed=1, swap=swap, opponent="strategic_v4", beta=1.0, teacher="v17-teach", student=None, shard=None)
    rows, summary = play_labelled_match(_FakeEnv(), task, None)
    assert summary["rows"] == len(rows["labels"]) > 0 and summary["result"] == (-1 if swap else 1)
    stats = summarize_matches([summary])
    assert stats["teacher_frac"] == 1.0 and "agree/all" not in stats  # no student was asked
    assert stats["units_hunter"] == 0 and stats["units_collector"] == 5  # everyone starts as a Collector
    assert stats["label_valid_frac"] > 0 and "first_hunter1_s" not in stats
    assert rows["graphic"].dtype == np.uint8 and (rows["agent_states"][:, :5, 2] > 0).all()
    assert rows["teacher_drove"].all() and (rows["student"] == STAND).all()


def test_mixed_control_needs_and_uses_the_student():
    task = MatchTask(seed=2, swap=False, opponent="mixture", beta=0.5, teacher="v17-teach", student="x", shard=None)
    with pytest.raises(ValueError):
        play_labelled_match(_FakeEnv(), task, None)
    rows, summary = play_labelled_match(_FakeEnv(length=8), task, MyPolicy(MyModel().eval()))
    stats = summarize_matches([summary, summary])
    assert 0.0 < stats["teacher_frac"] < 1.0
    assert 0.0 <= stats["agree/all"] <= 1.0 and stats["agree_share/role_collector"] == 1.0
    assert (rows["student"] < 8).all()  # asked every tick, whoever drove
    assert np.array_equal(rows["actions"][rows["teacher_drove"] == False], rows["student"][rows["teacher_drove"] == False])  # noqa: E712


def test_balanced_class_weights_give_each_present_class_equal_mass():
    classes = torch.zeros(1, 5, 3)
    classes[0, :3, 1] = 1  # three Hunters
    classes[0, 3, 2] = 1   # one Carrier
    classes[0, 4, 0] = 1   # one Collector
    weights = class_weights(classes, torch.ones(1, 5))
    per_class = (classes[0] * weights[0, :, None]).sum(0)
    assert torch.allclose(per_class, torch.full((3,), 5 / 3)) and torch.isclose(weights.sum(), torch.tensor(5.0))


def test_collection_follows_the_replay_ratio_after_warm_up():
    from blackout_env.train.distill.run import beta_for, parse_args, wants_matches

    args = parse_args(["--min-rows", "100", "--replay-ratio", "2", "--beta-decay", "0.5"])
    assert wants_matches(0, 0, 50, args)  # warm-up: always
    assert not wants_matches(150, 100, 100, args) and wants_matches(200, 100, 100, args)
    assert [beta_for(v, args) for v in range(4)] == [1.0, 1.0, 0.5, 0.25]


def test_shards_keep_analysis_fields_and_meta_and_the_ring_tracks_versions(tmp_path):
    rows = _rows(3, 0)
    rows["actions"] = np.zeros((3, 5), np.uint8)
    rows["kinds"][:, 0] = KIND_INDEX["hunt"]
    save_shard(tmp_path / "a.npz", rows, meta={"student_version": 4, "tag": "eval"})
    with np.load(tmp_path / "a.npz") as shard:
        assert "actions" in shard and shard_meta(shard)["student_version"] == 4
    dataset = DistillDataset(tmp_path, capacity=8)
    (loaded, meta), = list(dataset.new_shards())
    dataset.append(loaded, meta["student_version"])
    ring = dataset.composition(current_version=6)
    assert ring["ring/mean_version_age"] == 2 and ring["ring/teacher_only_frac"] == 0
    assert ring["ring/kind_share_hunt"] == 1.0


def test_label_accuracy_splits_by_role_and_kind():
    logits = torch.zeros(1, 5, 8)
    logits[0, :, 0] = 1  # always picks direction 0
    masks = torch.zeros(1, 5, 8, dtype=torch.bool)
    masks[0, 0, 0] = masks[0, 1, 3] = True  # unit 0 right, unit 1 wrong, others unlabelled
    classes = torch.zeros(1, 5, 3)
    classes[0, 0, 1] = classes[0, 1, 0] = 1  # unit 0 Hunter, unit 1 Collector
    kinds = torch.tensor([[KIND_INDEX["hunt"], KIND_INDEX["battery"], 0, 0, 0]])
    accuracy = LabelAccuracy(torch.device("cpu"))
    accuracy.add(logits, masks, classes, kinds)
    stats = accuracy.pop("fresh")
    assert stats["fresh/accuracy"] == 0.5 and stats["fresh/labels"] == 2
    assert stats["fresh/accuracy_role_hunter"] == 1.0 and stats["fresh/accuracy_role_collector"] == 0.0
    assert stats["fresh/accuracy_kind_hunt"] == 1.0 and stats["fresh/share_kind_battery"] == 0.5
    assert accuracy.pop("fresh") == {}  # popped


def test_inference_modes_are_deterministic_where_promised():
    from blackout_env.train.distill.inference import make_student

    torch.manual_seed(0)
    net = MyModel()
    team = {name: _view(False) for name in all_agents()[:5]}
    fixed = make_student(net, "fixed_tau")
    first, again = fixed.action_values(team), fixed.action_values(team)
    assert all(np.array_equal(first[a], again[a]) for a in first)
    moves = make_student(net, "tie_mean").act(team)
    assert all(abs(np.linalg.norm(v) - 1) < 1e-5 for v in moves.values())
    with pytest.raises(ValueError):
        make_student(net, "nope")
