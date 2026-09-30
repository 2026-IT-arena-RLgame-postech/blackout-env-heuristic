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


def test_composition_is_also_reported_by_time_of_play():
    task = MatchTask(seed=1, swap=False, opponent="strategic_v4", beta=1.0, teacher="v17-teach", student=None, shard=None)
    _, summary = play_labelled_match(_FakeEnv(), task, None)
    stats = summarize_matches([summary])
    # _FakeEnv's time_left stays 1.0: every tick is in the first bucket
    assert stats["units_collector_0_30s"] == 5 and "units_hunter_30_60s" not in stats


def test_collection_seeds_are_random_reproducible_and_avoid_eval_seeds():
    from blackout_env.train.distill.run import HELDOUT_SEEDS, collect_seed

    seeds = [collect_seed(7, n, reserved={123}) for n in range(500)]
    assert seeds == [collect_seed(7, n, reserved={123}) for n in range(500)]  # resume replays them
    assert len(set(seeds)) == 500 and not any(s in HELDOUT_SEEDS or s == 123 for s in seeds)
    assert seeds[:3] != [collect_seed(8, n, reserved=set()) for n in range(3)]


def test_opponent_pool_draws_by_weight_and_labels_categories():
    from blackout_env.train.distill.opponents import draw_opponent, opponent_category, parse_pool

    pool = parse_pool(["mixture=3", "checkpoints=1"])
    assert pool == {"mixture": 0.75, "checkpoints": 0.25}
    rng = np.random.default_rng(0)
    draws = [draw_opponent(rng, pool, ["a.pt", "b.pt"]) for _ in range(2000)]
    assert 0.2 < np.mean([d.startswith("ckpt:") for d in draws]) < 0.3
    assert {d for d in draws if d.startswith("ckpt:")} == {"ckpt:a.pt", "ckpt:b.pt"}
    with pytest.raises(ValueError):
        parse_pool(["nonsense"])
    with pytest.raises(ValueError):
        draw_opponent(rng, {"checkpoints": 1.0}, [])
    assert [opponent_category(x) for x in ("ckpt:r/final", "mixture:strategic_v18", "mixture:strategic_v4", "strategic_v17")] \
        == ["checkpoint", "top_tier", "heuristic", "top_tier"]


def test_ood_opponent_descriptors_build_and_fail_early():
    from blackout_env.train.distill.opponents import Idle, RandomWalker, check_opponent, make_opponent, opponent_name

    camp = make_opponent("strategic_v19:hunter_quota=4,hunter_mode=camp,sentry=false", 0)
    assert (camp.hunter_quota, camp.hunter_mode, camp.sentry) == (4, "camp", False)
    assert make_opponent("strategic_v17:steal_weight=3.0", 0).steal_weight == 3.0
    assert make_opponent("teacher:v21-teach", 0).act(_obs := {n: _view(False) for n in all_agents()[:5]}).keys() == _obs.keys()
    assert all((a == 0).all() for a in Idle().act(_obs).values())
    walker = RandomWalker(3)
    first = walker.act(_obs)
    assert all(np.isclose(np.linalg.norm(a), 1.0) for a in first.values())
    walker.reset()
    assert all(np.array_equal(first[n], a) for n, a in walker.act(_obs).items())  # seeded
    for bad in ("strategic_v19:hunter_mode=sit", "strategic_v99", "strategic_v17:quota", "ckpt:missing.pt", "teacher:none"):
        with pytest.raises((ValueError, KeyError)):
            check_opponent(bad)
    check_opponent("self")
    assert opponent_name("ckpt:checkpoints/offline/20260919-010456_42178/final.pt") == "ckpt-20260919-010456-final"
    assert opponent_name("strategic_v19:hunter_quota=4,hunter_mode=camp") == "strategic_v19-hunter_quota=4-hunter_mode=camp"


def test_self_opponent_mirrors_the_learner():
    task = MatchTask(seed=1, swap=False, opponent="self", beta=0.0, teacher="v17-teach", student="x", shard=None)
    _, summary = play_labelled_match(_FakeEnv(length=8), task, MyPolicy(MyModel().eval()))
    assert summary["opponent_played"] == "self"
    teacher_task = MatchTask(seed=1, swap=True, opponent="self", beta=1.0, teacher="v17-teach", student=None, shard=None)
    assert play_labelled_match(_FakeEnv(), teacher_task, None)[1]["rows"] > 0


def test_q_units_computes_only_the_first_rows_exactly():
    net = MyModel().eval()
    g, ts, ag = torch.rand(3, 13, 24, 24).round(), torch.rand(3, 4), torch.rand(3, 10, 12) * 2 - 1
    tau = torch.rand(3, 8)
    full = net(g, ts, ag, tau=tau)[0]
    own = net(g, ts, ag, tau=tau, q_units=5)[0]
    assert own.shape == (3, 5, 8) and torch.allclose(full[:, :5], own, atol=1e-6)


@pytest.mark.parametrize("team_b", [False, True])
def test_fast_student_matches_eager_mymodel(team_b):
    from blackout_env.train.distill.inference import fixed_tau, make_student

    class _EagerFixedTau(torch.nn.Module):  # the pre-trace path: all 10 rows, eager
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, graphic, team_state, agent_states, **_):
            return self.net(graphic, team_state, agent_states, tau=fixed_tau().expand(graphic.shape[0], -1))

    torch.manual_seed(0)
    net = MyModel().eval()
    names = all_agents()[5:] if team_b else all_agents()[:5]
    team = {name: _view(team_b) for name in names}
    fast = make_student(net, "fixed_tau").action_values(team)
    fast_again = make_student(net, "fixed_tau").action_values(team)  # traced on first use, then reused
    eager = MyPolicy(_EagerFixedTau(net)).action_values(team)
    for agent in names:
        assert np.allclose(fast[agent], eager[agent], atol=1e-5) and np.allclose(fast[agent], fast_again[agent])
