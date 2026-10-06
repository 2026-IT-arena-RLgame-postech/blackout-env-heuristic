"""P11: the continuous-angle head -- shard angles and their team frame, old shards, the zero-init
warm start, the offset loss, inference along offset headings, and the angle metrics."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from blackout_env.env.constants import all_agents
from blackout_env.heuristics.teach import TeachLabel
from blackout_env.model.direction_bins import (
    angle_bin, angle_residual, arc_mask, bin_vectors, mirror_angle, mirror_bin, move_angle, offset_vectors, unpack_bits,
    vector_to_bin, wrap_angle,
)
from blackout_env.model.my_model import MyModel
from blackout_env.model.my_policy import MyPolicy
from blackout_env.train.distill.collect import MatchTask, _labelled_row, play_labelled_match
from blackout_env.train.distill.dataset import ANGLES, DistillDataset, save_shard
from blackout_env.train.distill.loss import angle_offset_loss
from blackout_env.train.distill.metrics import LabelAccuracy, summarize_matches

from tests.test_distill import _FakeEnv, _rows, _view

K = 32
P10 = Path("/Users/mac/project/26rl/blackout-env/checkpoints/distill/frozen/p10_v22.pt")


def _heading(rad):
    return np.array([math.cos(rad), math.sin(rad)], np.float32)


def _p10_like(seed=0) -> MyModel:
    """A logits K = 32 net with a spread-out head (what P10 is), no angle head."""
    torch.manual_seed(seed)
    net = MyModel(n_actions=K, policy_head="logits").eval()
    with torch.no_grad():
        net.q_head.value_head[1].weight.mul_(20.0)
    return net


def _angle_net(seed=0, bias=None) -> MyModel:
    from blackout_env.train.distill.run import load_weights

    net = MyModel(n_actions=K, policy_head="logits", angle_head=True).eval()
    load_weights(net, _p10_like(seed).state_dict())
    if bias is not None:
        with torch.no_grad():
            net.q_head.angle_out.bias.copy_(torch.as_tensor(bias, dtype=torch.float32))
    return net


def _frames(n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, 13, 24, 24, generator=g).round(), torch.rand(n, 4, generator=g),
            torch.rand(n, 10, 12, generator=g) * 2 - 1)


# ------------------------------------------------------------------------------ angles in shards


def test_angle_helpers_wrap_mirror_and_bin():
    rng = np.random.default_rng(0)
    angles = rng.uniform(-math.pi, math.pi, 500)
    vectors = np.stack([np.cos(angles), np.sin(angles)], -1).astype(np.float32)
    assert np.array_equal(angle_bin(angles, K), vector_to_bin(vectors, K))  # nearest bin, as the labels
    assert (np.abs(angle_residual(angles, K)) <= math.pi / K + 1e-9).all()
    mirrored = mirror_angle(angles)  # t -> 90 deg - t is the world x <-> y swap
    assert np.allclose(np.stack([np.cos(mirrored), np.sin(mirrored)], -1), vectors[:, ::-1], atol=1e-6)
    assert np.array_equal(angle_bin(mirrored, K), mirror_bin(angle_bin(angles, K), K))
    assert np.allclose(mirror_angle(mirror_angle(angles)), angles)
    assert math.isnan(move_angle(np.zeros(2))) and np.isnan(mirror_angle(np.nan))
    assert np.allclose(offset_vectors(np.zeros(K)), bin_vectors(K), atol=1e-7)


@pytest.mark.parametrize("team_b", [False, True])
def test_shard_angles_follow_the_label_frame_exactly(team_b):
    names = all_agents()[5:] if team_b else all_agents()[:5]
    obs = {name: _view(team_b=team_b) for name in names}
    degrees = [3.0, 97.3, -170.0, 44.0, None]  # the last unit stands (no label, no angle)
    moves, labels = {}, {}
    for name, deg in zip(names, degrees):
        move = np.zeros(2, np.float32) if deg is None else 2.0 * _heading(math.radians(deg))  # clipped to [-1, 1]
        moves[name] = move
        mask = arc_mask(np.clip(move, -1, 1), K, 1)
        labels[name] = TeachLabel(mask, bool(mask.any()), "act", None)
    student = {names[0]: _heading(0.5)}
    row = _labelled_row(obs, labels, moves, student, {n: True for n in names}, bins=K, teacher_moves=moves)
    for j, (name, deg) in enumerate(zip(names, degrees)):
        if deg is None:
            assert np.isnan(row[ANGLES][j]) and row["labels"][j] == 0
            continue
        world = move_angle(np.clip(moves[name], -1, 1))
        expect = mirror_angle(world) if team_b else world
        assert np.isclose(wrap_angle(row[ANGLES][j] - expect), 0.0, atol=1e-6)
        # the stored angle's bin is the centre bin of the stored (mirrored) label
        bits = np.flatnonzero(unpack_bits(row["labels"][j], K))
        centre = int(angle_bin(row[ANGLES][j], K))
        assert sorted(bits.tolist()) == sorted(((centre + np.arange(-1, 2)) % K).tolist())
    s = row["student_angles"]
    assert np.isclose(s[0], mirror_angle(0.5) if team_b else 0.5, atol=1e-6) and np.isnan(s[1:]).all()


def test_relabelling_teachers_store_no_angles():
    obs = {name: _view(team_b=False) for name in all_agents()[:5]}
    labels = {name: TeachLabel(np.eye(8, dtype=bool)[1], True, "battery", None) for name in obs}
    moves = {name: _heading(0.3) for name in obs}
    row = _labelled_row(obs, labels, moves, {}, {n: True for n in obs}, teacher_moves=moves)
    assert np.isnan(row[ANGLES]).all()
    assert np.isnan(_labelled_row(obs, labels, moves, {}, {n: True for n in obs})[ANGLES]).all()


def test_old_shards_load_with_nan_angles_and_new_ones_round_trip(tmp_path):
    old = _rows(4, 1)  # a shard written before P11: no angles field
    save_shard(tmp_path / "a.npz", old)
    new = _rows(3, 0)
    new[ANGLES] = np.tile(np.array([0.1, np.nan, -2.0, 3.0, np.nan], np.float32), (3, 1))
    new["tick"] = np.arange(3, dtype=np.int32)
    save_shard(tmp_path / "b.npz", new)
    ring = DistillDataset(tmp_path, capacity=16)
    assert ring.refresh() == 7
    batch = ring.sample(256, np.random.default_rng(0), labelled_only=False)
    angles = batch[ANGLES]
    assert angles.shape == (256, 5) and angles.dtype == np.float32
    from_old = np.isnan(angles).all(axis=1)
    assert from_old.any() and (~from_old).any()
    assert np.allclose(angles[~from_old][:, [0, 2, 3]], [0.1, -2.0, 3.0])
    windows = ring.sample_windows(4, 2, 1, np.random.default_rng(1))
    assert windows[ANGLES].shape == (4, 3, 5)
    assert np.isnan(windows[ANGLES][~windows["valid"]]).all()
    assert ring.composition(0)["ring/angle_unit_frac"] == pytest.approx(9 / 35)


# ------------------------------------------------------------------------------ model and warm start


def test_angle_head_zero_init_equals_p10_at_step_zero():
    from blackout_env.train.distill.inference import INFERENCE, load_net, make_student

    p10, net = _p10_like(), _angle_net()
    g, ts, ag = _frames()
    with torch.no_grad():
        before = p10(g, ts, ag)[0]
        plain = net(g, ts, ag)
        with_angle = net(g, ts, ag, angle=True)
    assert torch.equal(plain[0], before) and torch.equal(with_angle[0], before)
    assert len(plain) == 5 and len(with_angle) == 6 and torch.equal(with_angle[5], torch.zeros(6, 10, K))
    loaded = load_net(net.state_dict())  # the checkpoint records it in its keys
    assert loaded.angle_head and not load_net(p10.state_dict()).angle_head
    assert MyModel.from_state_dict(p10.state_dict(), angle_head=True).angle_head
    with pytest.raises(ValueError):
        MyModel.from_state_dict(net.state_dict(), angle_head=False)
    for team_b in (False, True):
        names = all_agents()[5:] if team_b else all_agents()[:5]
        team = {name: _view(team_b) for name in names}
        for inference in INFERENCE:
            a, b = make_student(p10, inference).act(team), make_student(loaded, inference).act(team)
            assert all(np.allclose(a[n], b[n], atol=1e-6) for n in names), inference


@pytest.mark.skipif(not P10.exists(), reason="needs the frozen P10 checkpoint")
def test_real_p10_checkpoint_warm_starts_identically():
    from blackout_env.train.distill.inference import load_net, make_student
    from blackout_env.train.distill.run import Learner, load_weights, parse_args

    state = torch.load(P10, map_location="cpu", weights_only=True)["policy_state"]
    learner = Learner(parse_args(["--build", "x", "--device", "cpu", "--policy-head", "logits", "--action-bins", "32",
                                  "--angle-head"]))
    load_weights(learner.net, state)
    p10, p11 = load_net(state), load_net(learner.net.state_dict())
    team = {name: _view(False) for name in all_agents()[:5]}
    a, b = make_student(p10, "fixed_tau").act(team), make_student(p11, "fixed_tau").act(team)
    assert all(np.array_equal(a[n], b[n]) for n in team)


def test_angle_head_offsets_stay_within_half_a_bin_and_the_sequence_matches_steps():
    net = MyModel(n_actions=K, policy_head="logits", angle_head=True, temporal=True).eval()
    with torch.no_grad():
        net.q_head.angle_out.weight.normal_(0, 1.0)
        net.q_head.angle_out.bias.fill_(5.0)
    g, ts, ag = _frames(6)
    frames = (g.view(2, 3, *g.shape[1:]), ts.view(2, 3, 4), ag.view(2, 3, 10, 12))
    with torch.no_grad():
        q_seq, _, _, off_seq, _ = net.forward_sequence(*frames, q_units=5, angle=True)
        memory = None
        for t in range(3):
            out = net(frames[0][:, t], frames[1][:, t], frames[2][:, t], q_units=5, memory=memory, angle=True)
            memory = out[-1]
            assert torch.allclose(out[5], off_seq[:, t], atol=1e-5) and torch.allclose(out[0], q_seq[:, t], atol=1e-4)
    assert (off_seq.abs() <= math.pi / K + 1e-6).all() and off_seq.abs().max() > 0.9 * math.pi / K
    with pytest.raises(ValueError):
        MyModel(n_actions=K, angle_head=True)  # IQN head: no angle head


# ------------------------------------------------------------------------------ loss and learner


def test_angle_loss_targets_the_residual_of_the_nearest_bin_and_masks_nan():
    half = math.pi / K
    angles = torch.tensor([[2 * math.pi * 5 / K + 0.5 * half, float("nan")], [-0.25 * half, 2 * math.pi * 31 / K]])
    offsets = torch.zeros(2, 2, K, requires_grad=True)
    loss, stats = angle_offset_loss(offsets, angles)
    assert loss.item() == pytest.approx((0.25 + 0.0625 + 0.0) / 3, rel=1e-5)
    assert stats["angle_rows"].item() == 3
    loss.backward()
    grad = offsets.grad
    assert grad[0, 1].abs().sum() == 0  # the NaN row: no gradient at all
    assert grad[0, 0, 5] < 0 and grad[0, 0].abs().sum() == grad[0, 0, 5].abs()  # only bin 5, pushed up
    assert grad[1, 0, 0] > 0  # -0.25 half widths around bin 0: pushed down
    exact = torch.zeros(2, 2, K)
    exact[0, 0, 5], exact[1, 0, 0] = 0.5 * half, -0.25 * half
    assert angle_offset_loss(exact, angles)[0].item() == pytest.approx(0.0, abs=1e-10)
    weighted = angle_offset_loss(torch.zeros(2, 2, K), angles, torch.tensor([[3.0, 1.0], [1.0, 1.0]]))[0]
    assert weighted.item() == pytest.approx((3 * 0.25 + 0.0625) / 5, rel=1e-5)
    assert angle_offset_loss(torch.zeros(1, 1, K), torch.tensor([[float("nan")]]))[0].item() == 0.0


def _batch(n=4):
    from blackout_env.model.direction_bins import pack_bits

    rng = np.random.default_rng(0)
    angles = rng.uniform(-math.pi, math.pi, (n, 5)).astype(np.float32)
    angles[:, 4] = np.nan
    masks = np.zeros((n, 5, K), bool)
    for i in range(n):
        for j in range(4):
            masks[i, j] = arc_mask(_heading(angles[i, j]), K, 1)
    team_state = np.zeros((n, 4), np.float32)
    team_state[:, 2] = np.where(np.arange(n) % 2 == 0, 1.0 - 5 / 420, 0.5)  # 5 s (opening) or 210 s of play
    rows = {"graphic": np.zeros((n, 24, 24, 13), np.uint8), "team_state": team_state,
            "agent_states": np.zeros((n, 10, 12), np.float32), "labels": pack_bits(masks),
            "kinds": np.full((n, 5), 12, np.uint8), ANGLES: angles}
    rows["agent_states"][..., 9] = 1
    return rows


def test_learner_trains_the_angle_head_and_reports_angle_error():
    from blackout_env.train.distill.run import Learner, parse_args

    with pytest.raises(SystemExit):
        parse_args(["--build", "x", "--angle-head"])  # IQN head
    argv = ["--build", "x", "--device", "cpu", "--policy-head", "logits", "--action-bins", "32", "--angle-head",
            "--lr", "3e-3", "--opening-weight", "3"]
    learner = Learner(parse_args(argv))
    rows = _batch()
    *_, aux = learner._tensors(rows)
    assert torch.allclose(aux["elapsed"], torch.tensor([5.0, 210.0, 5.0, 210.0]), atol=1e-3)
    weights = learner.row_weights(torch.ones(4, 5), torch.zeros(4, 5, 3), aux["elapsed"])
    assert torch.equal(weights[:, 0], torch.tensor([3.0, 1.0, 3.0, 1.0]))
    learner.step(rows)
    first = learner.pop_stats()
    assert first["train/angle_loss"] > 0 and "train/angle_err_deg/all" in first and "train/angle_err_deg/opening" in first
    assert first["train/angle_n/all"] == 16 and first["train/angle_n/opening"] == 8
    for _ in range(40):
        learner.step(rows)
    later = learner.pop_stats()
    assert later["train/angle_loss"] < first["train/angle_loss"]
    assert later["train/angle_res_err_deg/all"] < first["train/angle_res_err_deg/all"]
    assert learner.net.q_head.angle_out.weight.abs().sum() > 0
    old = {k: v for k, v in rows.items() if k != ANGLES}  # rows without angles: no angle loss, still trains
    learner.step(old)
    assert learner.pop_stats()["train/angle_loss"] == 0.0


def test_window_weight_stacks_with_the_opening_weight():
    from blackout_env.train.distill.run import Learner, parse_args

    argv = ["--build", "x", "--device", "cpu", "--policy-head", "logits", "--action-bins", "32", "--angle-head",
            "--opening-weight", "3", "--window-weight", "2", "--window-seconds", "4,6"]
    learner = Learner(parse_args(argv))
    *_, aux = learner._tensors(_batch())
    weights = learner.row_weights(torch.ones(4, 5), torch.zeros(4, 5, 3), aux["elapsed"])
    assert torch.equal(weights[:, 0], torch.tensor([6.0, 1.0, 6.0, 1.0]))
    learner = Learner(parse_args(argv[:-4] + ["--window-weight", "2", "--window-seconds", "100,300"]))
    weights = learner.row_weights(torch.ones(4, 5), torch.zeros(4, 5, 3), aux["elapsed"])
    assert torch.equal(weights[:, 0], torch.tensor([3.0, 2.0, 3.0, 2.0]))
    learner = Learner(parse_args(argv + ["--join-weight", "10", "--join-seconds", "4.5,5.5"]))  # P20: stacks with both
    weights = learner.row_weights(torch.ones(4, 5), torch.zeros(4, 5, 3), aux["elapsed"])
    assert torch.equal(weights[:, 0], torch.tensor([60.0, 1.0, 60.0, 1.0]))
    learner = Learner(parse_args(argv + ["--join-weight", "10"]))  # default window 3.4-4.0 s misses rows at 5 s
    weights = learner.row_weights(torch.ones(4, 5), torch.zeros(4, 5, 3), aux["elapsed"])
    assert torch.equal(weights[:, 0], torch.tensor([6.0, 1.0, 6.0, 1.0]))


def test_stand_seconds_limits_stand_targets_to_its_window():
    from blackout_env.train.distill.run import Learner, parse_args

    base = ["--build", "x", "--device", "cpu", "--policy-head", "logits", "--action-bins", "32", "--angle-head",
            "--stand-action"]
    rows = _batch()  # unit 4 has no label; rows 0, 2 at 5 s of play, rows 1, 3 at 210 s
    _, _, _, masks, *_ = Learner(parse_args(base))._tensors(rows)
    assert masks[:, 4, K].tolist() == [True] * 4 and not masks[:, :4, K].any()
    _, _, _, masks, *_ = Learner(parse_args(base + ["--stand-seconds", "2.5,6"]))._tensors(rows)
    assert masks[:, 4, K].tolist() == [True, False, True, False] and not masks[:, :4, K].any()
    assert not masks[1, 4].any()  # outside the window: no target at all (as without --stand-action)


def test_label_accuracy_measures_bin_centres_without_offsets():
    acc = LabelAccuracy(torch.device("cpu"))
    logits = torch.zeros(1, 1, K)
    logits[0, 0, 3] = 5.0
    angle = 2 * math.pi * 3 / K + math.radians(1.5)
    masks = torch.zeros(1, 1, K, dtype=torch.bool)
    masks[0, 0, 3] = True
    acc.add(logits, masks, torch.tensor([[[1.0, 0, 0]]]), torch.ones(1, 1, dtype=torch.long),
            torch.tensor([[angle]]), None, torch.tensor([3.0]))
    out = acc.pop("x")
    assert out["x/angle_err_deg/all"] == pytest.approx(1.5, abs=1e-4) and out["x/angle_within2/opening"] == 1.0
    assert out["x/angle_bin_hit/all"] == 1.0 and out["x/accuracy"] == 1.0
    offsets = torch.zeros(1, 1, K)
    offsets[0, 0, 3] = math.radians(1.5)
    acc.add(logits, masks, torch.tensor([[[1.0, 0, 0]]]), torch.ones(1, 1, dtype=torch.long),
            torch.tensor([[angle]]), offsets, torch.tensor([30.0]))
    out = acc.pop("x")
    assert out["x/angle_err_deg/all"] == pytest.approx(0.0, abs=1e-4) and "x/angle_err_deg/opening" not in out


# ------------------------------------------------------------------------------ inference


@pytest.mark.parametrize("team_b", [False, True])
def test_angle_student_moves_off_the_bin_centres(team_b):
    from blackout_env.train.distill.inference import INFERENCE, make_student

    bias = np.linspace(-3, 3, K)
    net = _angle_net(bias=bias)
    names = all_agents()[5:] if team_b else all_agents()[:5]
    team = {name: _view(team_b) for name in names}
    student = make_student(net, "fixed_tau")
    moves = student.act(team)
    values, offsets = student.action_values(team), student.last_offsets
    for name in names:
        best = int(np.argmax(values[name]))
        assert np.allclose(moves[name], offset_vectors(offsets[name])[best], atol=1e-6)
        centre = bin_vectors(K)[best]
        assert float(moves[name] @ centre) < 1 - 1e-4  # not the bin centre
        assert abs(wrap_angle(move_angle(moves[name]) - 2 * math.pi * best / K)) <= math.pi / K + 1e-6
    # the world-frame move is the canonical one mirrored for team B: the canonical arg max bin
    # plus its offset, reflected
    from blackout_env.env.team_frame import canonical_obs

    view = canonical_obs(next(iter(team.values())))
    g = torch.tensor(view["graphic"]).permute(2, 0, 1)[None].float()
    with torch.no_grad():
        q, *_, off = net(g, torch.tensor(view["team_state"])[None], torch.tensor(view["agent_states"])[None], angle=True)
    from blackout_env.env.constants import unit_index
    from blackout_env.env.team_frame import canonical_unit_row

    for name in names:
        r = canonical_unit_row(unit_index(name), team_b)
        b = int(q[0, r].argmax())
        canonical = 2 * math.pi * b / K + float(off[0, r, b])
        expect = mirror_angle(canonical) if team_b else canonical
        assert abs(wrap_angle(move_angle(moves[name]) - expect)) < 1e-5
    for inference in INFERENCE:  # every mode moves, along offset headings
        for move in make_student(net, inference).act(team).values():
            assert np.isclose(np.linalg.norm(move), 1.0, atol=1e-5)


def test_eager_mymodel_policy_uses_offsets_too():
    net = _angle_net(bias=np.full(K, 1.0))
    team = {name: _view(False) for name in all_agents()[:5]}
    eager = MyPolicy(net).act(team)
    from blackout_env.train.distill.inference import make_student

    fast = make_student(net, "fixed_tau").act(team)
    assert all(np.allclose(eager[n], fast[n], atol=1e-5) for n in team)
    assert all(abs(wrap_angle(move_angle(eager[n]) - 2 * math.pi * int(vector_to_bin(eager[n], K)) / K)) > 1e-3 for n in team)


def test_labelled_match_records_angles_and_the_angle_error():
    task = MatchTask(seed=3, swap=True, opponent="strategic_v4", beta=0.0, teacher="act:strategic_v21_anyangle",
                     student="x", shard=None, action_bins=K, act_arc_bins=1, tag="eval")
    rows, summary = play_labelled_match(_FakeEnv(length=8), task, make_angle_student())
    labelled = rows["labels"] != 0
    assert np.isfinite(rows[ANGLES][labelled]).all() and np.isnan(rows[ANGLES][~labelled]).all()
    stats = summarize_matches([summary])
    assert 0.0 <= stats["angle_err_deg/all"] <= 180.0 and 0.0 <= stats["angle_within2/all"] <= 1.0
    assert "angle_err_deg/opening" in stats  # the fake match sits at 0 s of play
    assert 0.0 <= stats["agree/all"] <= 1.0


def make_angle_student():
    from blackout_env.train.distill.inference import make_student

    return make_student(_angle_net(bias=np.linspace(-1, 1, K)), "fixed_tau")


# ------------------------------------------------------------------------------ --angle-arc-bins


def test_arc_offsets_reach_the_neighbouring_bins_and_the_range_is_in_the_checkpoint():
    from blackout_env.train.distill.inference import load_net

    net = MyModel(n_actions=K, policy_head="logits", angle_head=True, angle_arc_bins=1).eval()
    with torch.no_grad():
        net.q_head.angle_out.bias.fill_(10.0)
        off = net(*_frames(2), angle=True)[5]
    assert torch.allclose(off, torch.full_like(off, 1.5 * 2 * math.pi / K), atol=1e-5)
    loaded = load_net(net.state_dict())
    assert loaded.angle_arc_bins == 1 and torch.equal(loaded.q_head.angle_scale, net.q_head.angle_scale)
    assert load_net(_angle_net().state_dict()).angle_arc_bins == 0


def test_arc_loss_trains_every_bin_of_the_arc_toward_the_same_angle():
    half = math.pi / K
    angle = 2 * math.pi * 5 / K + 0.5 * half
    angles = torch.tensor([[angle]])
    exact = torch.zeros(1, 1, K)
    for b in (4, 5, 6):
        exact[0, 0, b] = wrap_angle(angle - 2 * math.pi * b / K)
    assert angle_offset_loss(exact, angles, arc_bins=1)[0].item() == pytest.approx(0.0, abs=1e-10)
    assert angle_offset_loss(exact, angles)[0].item() == pytest.approx(0.0, abs=1e-10)  # the centre alone
    offsets = torch.zeros(1, 1, K, requires_grad=True)
    loss, stats = angle_offset_loss(offsets, angles, arc_bins=1)
    # bins 4, 5, 6 see residuals 2.5, 0.5, -1.5 half widths
    assert loss.item() == pytest.approx((6.25 + 0.25 + 2.25) / 3, rel=1e-5)
    assert stats["angle_loss_centre"].item() == pytest.approx(0.25, rel=1e-5)
    loss.backward()
    nonzero = torch.nonzero(offsets.grad[0, 0]).flatten().tolist()
    assert nonzero == [4, 5, 6] and offsets.grad[0, 0, 6] > 0 > offsets.grad[0, 0, 4]


def test_arc_learner_warm_starts_like_p10_and_trains():
    from blackout_env.train.distill.run import Learner, load_weights, parse_args

    learner = Learner(parse_args(["--build", "x", "--device", "cpu", "--policy-head", "logits", "--action-bins", "32",
                                  "--angle-head", "--angle-arc-bins", "1", "--lr", "3e-3"]))
    p10 = _p10_like()
    load_weights(learner.net, p10.state_dict())
    g, ts, ag = _frames()
    with torch.no_grad():
        out = learner.net.eval()(g, ts, ag, angle=True)
    assert torch.equal(out[0], p10(g, ts, ag)[0]) and not out[5].any()
    rows = _batch()
    learner.step(rows)
    first = learner.pop_stats()["train/angle_loss"]
    for _ in range(30):
        learner.step(rows)
    assert learner.pop_stats()["train/angle_loss"] < first
