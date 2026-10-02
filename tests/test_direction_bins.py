"""P4: K direction bins (model/direction_bins.py, --action-bins) -- labels, warm start, shards, inference."""

import math

import numpy as np
import pytest
import torch

from blackout_env.env.constants import all_agents
from blackout_env.env.team_frame import DIR_BITS_MIRROR, DIR_MIRROR
from blackout_env.heuristics.teach import TeachLabel
from blackout_env.model.direction_bins import (
    arc_mask, bin_vectors, bins_of_width, check_bins, convert_bits, mirror_bin, mirror_bits, pack_bits, unpack_bits,
    vector_to_bin, widen_mask,
)
from blackout_env.model.my_model import MyModel
from blackout_env.model.my_policy import DIRECTION_VECTORS, MyPolicy, direction_vector_to_idx
from blackout_env.train.distill.act_teacher import act_label
from blackout_env.train.distill.collect import MatchTask, _labelled_row, bin_labels, play_labelled_match
from blackout_env.train.distill.dataset import DistillDataset, save_shard, unpack_labels
from blackout_env.train.distill.metrics import STAND, summarize_matches

from tests.test_distill import _FakeEnv, _rows, _view


def _heading(deg):
    return np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))], np.float32)


# ------------------------------------------------------------------------------ K = 8 is the compass


def test_eight_bins_are_the_compass_bit_for_bit():
    assert np.array_equal(bin_vectors(8).view(np.uint32), DIRECTION_VECTORS.view(np.uint32))
    moves = np.random.default_rng(0).normal(size=(2000, 2)).astype(np.float32)
    moves[:8] = DIRECTION_VECTORS  # exact compass moves, and the zero move
    moves[8] = 0
    assert np.array_equal(vector_to_bin(moves, 8), direction_vector_to_idx(moves))
    assert np.array_equal(mirror_bin(np.arange(8), 8), DIR_MIRROR)
    every = np.arange(256, dtype=np.uint8)
    assert np.array_equal(mirror_bits(every, 8), DIR_BITS_MIRROR)
    masks = unpack_bits(every, 8)
    assert np.array_equal(masks, unpack_labels(every)) and np.array_equal(pack_bits(masks), every)
    assert pack_bits(masks).dtype == np.uint8
    assert np.array_equal(widen_mask(masks, 8), masks)  # +-22.5 deg at 45 deg spacing: the direction alone


def test_bin_labels_at_eight_bins_equal_the_teachers_labels():
    rng = np.random.default_rng(1)
    task = MatchTask(seed=0, swap=False, opponent="x", beta=1.0, teacher="v21-teach", student=None, shard=None)
    assert task.bins_are_compass
    labels, moves = {}, {}
    for i, name in enumerate(all_agents()[:5] * 40):
        name = f"{name}_{i}"
        move = rng.normal(size=2).astype(np.float32) * (3.0 if i % 3 else 0.5)
        if i % 2:
            labels[name], moves[name] = act_label(move), move  # an act teacher's label of its raw move
        else:
            mask = rng.random(8) < 0.3
            labels[name] = TeachLabel(mask, bool(mask.any()), "battery", None)
    binned = bin_labels(labels, moves, task)  # the K-bin path forced at K = 8
    assert all(np.array_equal(binned[n].mask, labels[n].mask) for n in labels)


def test_eight_bin_run_args_and_model_are_unchanged():
    from blackout_env.train.distill.run import Learner, parse_args

    args = parse_args(["--build", "x", "--device", "cpu"])
    assert (args.action_bins, args.act_arc_bins, args.label_width_deg, args.inference) == (8, 0, 22.5, "iqn")
    learner = Learner(args)
    assert learner.net.q_head.value_head[1].out_features == 8
    task = MatchTask(seed=1, swap=True, opponent="strategic_v4", beta=1.0, teacher="v17-teach", student=None, shard=None)
    rows, summary = play_labelled_match(_FakeEnv(), task, None)
    assert rows["labels"].dtype == np.uint8 and summary["action_bins"] == 8
    assert not any(k.startswith("agree8") for k in summary["counts"])


# ------------------------------------------------------------------------------ labels


def test_set_labels_widen_to_the_bins_within_the_half_width():
    mask = np.zeros(8, bool)
    mask[0] = True
    assert np.flatnonzero(widen_mask(mask, 32)).tolist() == [0, 1, 2, 30, 31]  # +-22.5 deg inclusive
    assert np.flatnonzero(widen_mask(mask, 32, 11.25)).tolist() == [0, 1, 31]
    mask[1] = True  # 0 and 45 deg: the boundary bin at 22.5 deg belongs to both
    assert np.flatnonzero(widen_mask(mask, 32)).tolist() == [0, 1, 2, 3, 4, 5, 6, 30, 31]
    assert np.flatnonzero(widen_mask(mask, 16)).tolist() == [0, 1, 2, 3, 15]
    assert not widen_mask(np.zeros(8, bool), 64).any()


def test_act_labels_take_the_raw_heading_not_its_compass_snap():
    task = MatchTask(seed=0, swap=False, opponent="x", beta=1.0, teacher="act:strategic_v21_anyangle", student=None,
                     shard=None, action_bins=32, act_arc_bins=1)
    assert not task.bins_are_compass
    move = 0.9 * _heading(30.0)  # snaps to 45 deg on the compass; bin 3 (33.75 deg) at K = 32
    labels = {"a": act_label(move), "b": act_label(np.zeros(2, np.float32)), "c": act_label(np.array([3.0, 1.0], np.float32))}
    binned = bin_labels(labels, {"a": move, "b": np.zeros(2, np.float32), "c": np.array([3.0, 1.0], np.float32)}, task)
    assert np.flatnonzero(binned["a"].mask).tolist() == [2, 3, 4] and binned["a"].valid
    assert not binned["b"].valid and not binned["b"].mask.any() and binned["b"].mask.shape == (32,)
    assert np.flatnonzero(binned["c"].mask).tolist() == [3, 4, 5]  # clipped to (1, 1) first, as the env does
    assert np.flatnonzero(arc_mask(_heading(30.0), 32)).tolist() == [3]
    assert np.flatnonzero(arc_mask(_heading(-1.0), 64, 2)).tolist() == [0, 1, 2, 62, 63]


@pytest.mark.parametrize("k", [16, 32, 64])
def test_bit_sets_round_trip_and_mirror_like_the_moves(k):
    rng = np.random.default_rng(k)
    masks = rng.random((50, 5, k)) < 0.2
    masks[0, 0, k - 1] = True  # bit 63 at K = 64
    bits = pack_bits(masks)
    assert bits.dtype == np.uint64 and np.array_equal(unpack_bits(bits, k), masks)
    # the mirror of a heading's bin is the bin of the mirrored heading (x <-> y)
    for deg in np.arange(0, 360, 360 / k):
        b = int(vector_to_bin(_heading(deg), k))
        assert mirror_bin(b, k) == int(vector_to_bin(_heading(deg)[::-1].copy(), k))
    single = pack_bits(np.eye(k, dtype=bool))
    assert np.array_equal(np.log2(mirror_bits(single, k).astype(np.float64)).astype(int), mirror_bin(np.arange(k), k))


def test_team_b_rows_mirror_k_bin_labels_and_moves():
    obs = {name: _view(team_b=True) for name in all_agents()[5:]}
    labels, moves = {}, {}
    for j, name in enumerate(obs):
        mask = np.zeros(32, bool)
        mask[3 * j + 1] = True
        labels[name] = TeachLabel(mask, True, "act", None)
        moves[name] = bin_vectors(32)[3 * j + 1]
    row = _labelled_row(obs, labels, moves, {}, {name: True for name in obs}, bins=32)
    assert row["labels"].dtype == np.uint64
    for j in range(5):
        assert np.flatnonzero(unpack_bits(row["labels"][j], 32)).tolist() == [mirror_bin(3 * j + 1, 32)]
        assert row["actions"][j] == mirror_bin(3 * j + 1, 32)
    assert (row["student"] == STAND).all()


def test_convert_bits_widens_old_shards_and_keeps_same_k():
    every = np.arange(256, dtype=np.uint8)
    assert np.array_equal(unpack_bits(convert_bits(every, 8, 32), 32), widen_mask(unpack_bits(every, 8), 32))
    bits = pack_bits(np.random.default_rng(0).random((10, 5, 32)) < 0.3)
    assert convert_bits(bits, 32, 32) is bits
    one = pack_bits(np.eye(32, dtype=bool)[[0, 1, 2, 3]])  # 0, 11.25, 22.5, 33.75 deg -> the compass
    assert [np.flatnonzero(m).tolist() for m in unpack_bits(convert_bits(one, 32, 8), 8)] == [[0], [0], [0, 1], [1]]
    assert convert_bits(one, 32, 8).dtype == np.uint8


def test_uint8_shards_load_into_a_k_bin_ring(tmp_path):
    old = _rows(3, 1)  # labels 1, 2, 3 = directions {0}, {1}, {0, 1}: an 8-way shard as before P4
    save_shard(tmp_path / "a.npz", old)
    new = _rows(2, 0)
    new["labels"] = pack_bits(np.eye(32, dtype=bool)[[5, 9]])[:, None].repeat(5, 1)
    save_shard(tmp_path / "b.npz", new, meta={"action_bins": 32})
    ring = DistillDataset(tmp_path, capacity=8, action_bins=32)
    assert ring.refresh() == 5
    labels = ring.sample(64, np.random.default_rng(0), labelled_only=False)["labels"]
    assert labels.dtype == np.uint64
    got = {tuple(np.flatnonzero(unpack_bits(b, 32))) for b in labels[:, 0]}
    assert got == {(0, 1, 2, 30, 31), (2, 3, 4, 5, 6), (0, 1, 2, 3, 4, 5, 6, 30, 31), (5,), (9,)}
    eight = DistillDataset(tmp_path, capacity=8)  # no action_bins: stored as is, the old ring
    (rows, _), _ = list(eight.new_shards())
    assert rows["labels"].dtype == np.uint8 and np.array_equal(rows["labels"], old["labels"])


# ------------------------------------------------------------------------------ warm start


@pytest.mark.parametrize("k, stand", [(16, False), (32, False), (64, False), (32, True)])
def test_k_bin_net_warm_started_from_an_eight_way_net_moves_identically(k, stand):
    from blackout_env.train.distill.inference import load_net, make_student
    from blackout_env.train.distill.run import load_weights

    torch.manual_seed(0)
    eight = MyModel().eval()
    with torch.no_grad():  # a trained-looking head: logits spread over several units
        eight.q_head.value_head[1].weight.mul_(20.0)
    net = MyModel(n_actions=k + int(stand)).eval()
    load_weights(net, eight.state_dict())
    assert bins_of_width(load_net(net.state_dict()).q_head.value_head[1].out_features) == (k, stand)
    step = k // 8
    g, ts, ag = torch.rand(16, 13, 24, 24).round(), torch.rand(16, 4), torch.rand(16, 10, 12) * 2 - 1
    tau = torch.rand(16, 8)
    with torch.no_grad():
        q8, qk = eight(g, ts, ag, tau=tau)[0], net(g, ts, ag, tau=tau)[0]
    assert torch.allclose(qk[..., :k:step], q8, atol=1e-5)  # the compass bins keep their logits
    assert torch.equal(qk[..., :k].argmax(-1), q8.argmax(-1) * step)  # and win wherever they did
    if stand:
        assert (qk[..., k] < q8.min(-1).values).all()
    for team_b in (False, True):  # the traced student plays the same moves
        names = all_agents()[5:] if team_b else all_agents()[:5]
        team = {name: _view(team_b) for name in names}
        before, after = make_student(eight, "fixed_tau").act(team), make_student(net, "fixed_tau").act(team)
        assert all(np.allclose(before[n], after[n], atol=1e-6) for n in names)


# ------------------------------------------------------------------------------ learner, inference, collection


@pytest.mark.parametrize("k, stand", [(32, False), (64, True)])
def test_learner_trains_on_k_bit_labels(k, stand):
    from blackout_env.train.distill.run import Learner, parse_args

    argv = ["--build", "x", "--device", "cpu", "--action-bins", str(k)] + (["--stand-action"] if stand else [])
    learner = Learner(parse_args(argv))
    masks = np.zeros((2, 5, k), bool)
    masks[0, 0, [k - 1, 0, 1]] = True
    masks[1, 2, 7] = True
    rows = {
        "graphic": np.zeros((2, 24, 24, 13), np.uint8), "team_state": np.zeros((2, 4), np.float32),
        "agent_states": np.zeros((2, 10, 12), np.float32), "labels": pack_bits(masks), "kinds": np.ones((2, 5), np.uint8),
    }
    rows["agent_states"][..., 9] = 1
    _, _, _, got, _, _, _ = learner._tensors(rows)
    assert got.shape == (2, 5, k + int(stand)) and torch.equal(got[..., :k], torch.from_numpy(masks))
    if stand:
        assert got[0, 1, k] and not got[0, 0, k]
    learner.step(rows)
    assert np.isfinite(learner.pop_stats()["train/ce"])


@pytest.mark.parametrize("team_b", [False, True])
def test_k_bin_student_values_and_moves(team_b):
    from blackout_env.train.distill.inference import INFERENCE, fixed_tau, make_student

    class _Eager(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, graphic, team_state, agent_states, **_):
            return self.net(graphic, team_state, agent_states, tau=fixed_tau().expand(graphic.shape[0], -1))

    torch.manual_seed(0)
    net = MyModel(n_actions=33).eval()
    names = all_agents()[5:] if team_b else all_agents()[:5]
    team = {name: _view(team_b) for name in names}
    fast = make_student(net, "fixed_tau").action_values(team)
    eager = MyPolicy(_Eager(net)).action_values(team)
    assert all(len(fast[n]) == 33 and np.allclose(fast[n], eager[n], atol=1e-5) for n in names)
    moves = make_student(net, "fixed_tau").act(team)
    for n in names:
        best = int(np.argmax(fast[n]))
        assert np.array_equal(moves[n], bin_vectors(32)[best] if best < 32 else np.zeros(2, np.float32))
    for inference in INFERENCE:
        for move in make_student(net, inference).act(team).values():
            assert np.isclose(np.linalg.norm(move), 1.0, atol=1e-5) or not move.any()


def test_arc_mean_averages_the_contiguous_top_arc_only():
    from blackout_env.train.distill.inference import arc_mean_move

    row = np.full(33, -10.0)
    row[[3, 4]] = 5.0  # a heading between bins 3 and 4
    row[20] = 5.0  # a second, separate peak: not part of the arc
    assert np.allclose(arc_mean_move(row), _heading(3.5 * 360 / 32), atol=1e-6)
    row[32] = 9.0
    assert not arc_mean_move(row).any()  # STAND
    eight = np.zeros(8)
    eight[0] = 1.0
    assert np.allclose(arc_mean_move(eight - 5 * (np.arange(8) > 0)), DIRECTION_VECTORS[0])


def test_k_bin_labelled_match_with_an_any_angle_act_teacher():
    task = MatchTask(seed=3, swap=True, opponent="strategic_v4", beta=0.5, teacher="act:strategic_v21_anyangle",
                     student="x", shard=None, action_bins=32, act_arc_bins=1)
    student = MyPolicy(MyModel(n_actions=32).eval())
    rows, summary = play_labelled_match(_FakeEnv(length=8), task, student)
    assert rows["labels"].dtype == np.uint64 and summary["action_bins"] == 32
    sets = unpack_bits(rows["labels"], 32)
    assert set(sets.sum(-1).ravel()) <= {0, 3}  # nearest bin +- 1, or no label (standing)
    assert ((rows["student"] < 32) | (rows["student"] == STAND)).all() and (rows["student"] < 32).any()
    stats = summarize_matches([summary])
    assert 0.0 <= stats["agree/all"] <= 1.0 and 0.0 <= stats["agree8/all"] <= 1.0
    assert "mean_turn_deg/student" in stats or "mean_turn_deg/teacher" in stats


def test_bins_must_be_multiples_of_eight():
    for bad in (4, 12, 128):
        with pytest.raises(ValueError):
            check_bins(bad)
    assert bins_of_width(9) == (8, True) and bins_of_width(64) == (64, False)
