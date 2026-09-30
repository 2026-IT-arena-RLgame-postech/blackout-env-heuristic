"""Act-teachers (any, possibly stateful, policy as a distillation teacher) and sequence data:
every full-window tick stored in order with its ``tick``, and ring windows for recurrent students."""

import numpy as np
import pytest

import blackout_env.train.distill.collect as collect
from blackout_env.env.constants import all_agents, team_a_agents, team_b_agents
from blackout_env.heuristics import make_heuristic
from blackout_env.model.base import BaseModel
from blackout_env.model.my_policy import DIRECTION_VECTORS
from blackout_env.train.distill.act_teacher import ActTeacher, act_label
from blackout_env.train.distill.collect import TEACHERS, MatchTask, play_labelled_match, teacher_arg
from blackout_env.train.distill.dataset import (
    FIELDS, KIND_INDEX, MATCH_TICKS, UNKNOWN_TICK, DistillDataset, save_shard, unpack_labels,
)
from tests.test_distill import _FakeEnv, _view


class _RecordingEnv(_FakeEnv):
    """_FakeEnv that logs every action it executes and drops ``missing`` from the obs on the
    ticks in ``partial`` (a partial decision window)."""

    def __init__(self, length=12, partial=(), missing=None):
        super().__init__(length)
        self.partial, self.missing, self.executed = set(partial), missing, []

    def _obs(self):
        obs = super()._obs()
        if self.t in self.partial:
            obs.pop(self.missing)
        return obs

    def step(self, actions):
        self.executed.append({name: np.array(a, np.float32) for name, a in actions.items()})
        return super().step(actions)


class _Counter(BaseModel):
    """Stateful: its move is a function of how many times it was called (and zero every 3rd)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.calls, self.seen = 0, []

    def act(self, obs):
        self.calls += 1
        self.seen.append(sorted(obs))
        k = self.calls
        return {name: (np.zeros(2, np.float32) if k % 3 == 0 else 0.5 * DIRECTION_VECTORS[(k + i) % 8])
                for i, name in enumerate(sorted(obs))}


def test_act_labels_are_the_snapped_move_and_invalid_when_standing():
    for k in range(8):
        label = act_label(0.3 * DIRECTION_VECTORS[k])
        assert label.valid and label.kind == "act" and np.flatnonzero(label.mask).tolist() == [k]
    angle = np.deg2rad(30)  # nearer 45 than 0
    assert np.flatnonzero(act_label(np.array([np.cos(angle), np.sin(angle)])).mask).tolist() == [1]
    assert np.flatnonzero(act_label(np.array([4.0, 1.2])).mask).tolist() == [1]  # clipped to (1, 1) first, like the env
    for still in (np.zeros(2), np.array([1e-7, -1e-7])):
        label = act_label(still)
        assert not label.valid and not label.mask.any()
    assert KIND_INDEX["stand"] == 11 and KIND_INDEX["act"] == 12  # appended: stored indices unchanged


def test_act_teacher_labels_equal_the_wrapped_heuristics_moves():
    obs = {name: _view(False) for name in team_a_agents()}
    teacher, reference = ActTeacher("strategic_v19"), make_heuristic("strategic_v19")
    teacher.reset(), reference.reset()
    for _ in range(6):
        labels, moves = teacher.teach(obs), reference.act(obs)
        for name in obs:
            assert np.array_equal(teacher.last_actions[name], np.asarray(moves[name], np.float32))
            assert labels[name].mask.tolist() == act_label(moves[name]).mask.tolist()
    assert ActTeacher("strategic_v17:steal_weight=3.0").policy.steal_weight == 3.0


def test_teacher_descriptors_resolve_everywhere_and_fail_early():
    assert "act:strategic_v19" in TEACHERS and "v21-teach" in TEACHERS and "nope" not in TEACHERS
    assert isinstance(TEACHERS["act:strategic_v19"](), ActTeacher)
    assert teacher_arg("act:strategic_v19:hunter_quota=4") == "act:strategic_v19:hunter_quota=4"
    assert teacher_arg("v21-teach") == "v21-teach"
    import argparse
    for bad in ("act:self", "act:", "act:strategic_v99", "v99-teach"):
        with pytest.raises(argparse.ArgumentTypeError):
            teacher_arg(bad)
    from blackout_env.train.distill.opponents import make_opponent
    assert isinstance(make_opponent("teacher:act:strategic_v4", 0), ActTeacher)  # also as an opponent
    from blackout_env.train.distill.run import parse_args
    assert parse_args(["--teacher", "act:strategic_v17"]).teacher == "act:strategic_v17"


@pytest.mark.parametrize("swap", [False, True])
def test_stateful_teacher_is_stepped_once_per_tick_including_partial_windows(monkeypatch, swap):
    learners = team_b_agents() if swap else team_a_agents()
    teachers = []
    monkeypatch.setitem(TEACHERS, "act:counter", lambda: teachers.append(_act(_Counter())) or teachers[-1])
    env = _RecordingEnv(length=12, partial={3, 7}, missing=learners[2])
    task = MatchTask(seed=1, swap=swap, opponent="idle", beta=1.0, teacher="act:counter", student=None, shard=None)
    rows, summary = play_labelled_match(env, task, None)
    counter = teachers[0].policy
    assert counter.calls == len(env.executed) == 12  # every decision tick, partial ones too
    assert counter.seen[3] == sorted(set(learners) - {learners[2]})
    assert rows["tick"].tolist() == [t for t in range(12) if t not in (3, 7)]  # gaps mark partial windows
    assert summary["rows"] == 10 and rows["tick"].dtype == np.int32
    # beta = 1 executes the counter's raw moves; its zero moves are unlabelled (STAND targets)
    reference = _Counter()
    for t, executed in enumerate(env.executed):
        moves = reference.act({name: None for name in learners if not (t in (3, 7) and name == learners[2])})
        for name, move in moves.items():
            assert np.array_equal(executed[name], move)
    stood = [t for t in range(12) if (t + 1) % 3 == 0 and t not in (3, 7)]
    recorded = rows["tick"].tolist()
    for t in stood:
        assert rows["labels"][recorded.index(t)].sum() == 0  # stored although nobody has a label
    assert (rows["kinds"][rows["labels"] > 0] == KIND_INDEX["act"]).all()


def _act(policy) -> ActTeacher:
    teacher = ActTeacher("idle")
    teacher.policy = policy
    return teacher


@pytest.mark.parametrize("swap", [False, True])
def test_beta_one_act_teacher_plays_exactly_the_heuristic(swap):
    learners = team_b_agents() if swap else team_a_agents()
    env = _RecordingEnv(length=15, partial={5}, missing=learners[0])
    task = MatchTask(seed=3, swap=swap, opponent="idle", beta=1.0, teacher="act:strategic_v19", student=None, shard=None)
    rows, _ = play_labelled_match(env, task, None)
    reference = make_heuristic("strategic_v19")
    reference.reset()
    replay = _FakeEnv(length=15)
    replay.reset()
    for t, executed in enumerate(env.executed):
        replay.t = t
        obs = {n: o for n, o in replay._obs().items() if n in learners and not (t == 5 and n == learners[0])}
        moves = reference.act(obs)
        assert all(np.array_equal(executed[n], np.asarray(moves[n], np.float32)) for n in obs)
    assert len(rows["tick"]) == 14


def test_unlabelled_rows_are_kept_in_shards_with_consecutive_ticks(tmp_path):
    task = MatchTask(seed=1, swap=False, opponent="idle", beta=1.0, teacher="act:idle", student=None, shard=None)
    rows, summary = play_labelled_match(_FakeEnv(length=9), task, None)
    assert summary["rows"] == 9 and not rows["labels"].any()  # idle: every unit stands, nothing labelled
    save_shard(tmp_path / "a.npz", rows, meta=summary)
    dataset = DistillDataset(tmp_path, capacity=100)
    assert dataset.refresh() == 9
    windows = dataset.sample_windows(4, length=5, burn_in=2, rng=np.random.default_rng(0))
    assert windows["valid"].any() and (np.diff(windows["tick"], axis=1)[windows["valid"][:, 1:] & windows["valid"][:, :-1]] == 1).all()
    with pytest.raises(RuntimeError):
        dataset.sample(4, np.random.default_rng(0))  # labelled-only sampling finds nothing
    assert len(dataset.sample(4, np.random.default_rng(0), labelled_only=False)["labels"]) == 4


def test_old_shards_without_tick_reconstruct_it_from_time_left(tmp_path):
    rows = _rows(5, ticks=None)
    rows["team_state"][:, 2] = 1.0 - (np.array([1, 2, 3, 5, 6]) / MATCH_TICKS).astype(np.float32)
    save_shard(tmp_path / "old.npz", rows)
    dataset = DistillDataset(tmp_path, capacity=10)
    dataset.refresh()
    assert dataset._tick[:5].tolist() == [1, 2, 3, 5, 6]
    dataset.append(_rows(3, ticks=None))  # time left not increasing: unknown
    assert (dataset._tick[5:8] == UNKNOWN_TICK).all()
    windows = dataset.sample_windows(64, length=3, burn_in=0, rng=np.random.default_rng(1))
    unknown = windows["valid"] & (windows["match"] == 1)
    assert (unknown.sum(1)[unknown.any(1)] == 1).all()  # unknown ticks: runs of one row


# ------------------------------------------------------------------------------ windows

def _rows(n, ticks, first_id=0, labelled=True):
    return {
        "graphic": np.zeros((n, 2, 2, 1), np.uint8), "team_state": np.zeros((n, 4), np.float32),
        "agent_states": (first_id + 1 + np.arange(n, dtype=np.float32))[:, None, None] * np.ones((1, 10, 12), np.float32),
        "labels": np.full((n, 5), 1 if labelled else 0, np.uint8), "kinds": np.zeros((n, 5), np.uint8),
        **({} if ticks is None else {"tick": np.asarray(ticks, np.int32)}),
    }


def _reference(logical, target, position, span, max_gap):
    """(valid, reset) of one window by walking the ring's logical rows [(match, tick), ...]."""
    first = target - position
    exists = lambda j: 0 <= j < len(logical)
    def linked(j):  # logical row j continues row j - 1
        if not (exists(j) and exists(j - 1)):
            return False
        (m0, t0), (m1, t1) = logical[j - 1], logical[j]
        return m0 == m1 and t0 >= 0 and 0 < t1 - t0 <= max_gap
    valid = np.zeros(span, bool)
    valid[position] = True
    for p in range(position - 1, -1, -1):
        if not linked(first + p + 1):
            break
        valid[p] = True
    for p in range(position + 1, span):
        if not linked(first + p):
            break
        valid[p] = True
    reset = np.array([valid[p] and not linked(first + p) for p in range(span)])
    return valid, reset


@pytest.mark.parametrize("max_gap", [1, 3])
def test_windows_stay_inside_one_run_and_flag_reset_valid_and_loss(max_gap):
    dataset = DistillDataset("unused", capacity=20)
    shards = [list(range(6)), list(range(7)), [0, 1, 2, 5, 6, 7], [0, 1, 2, 3]]  # 23 rows: wraps
    logical, next_id = [], 0
    for m, ticks in enumerate(shards):
        dataset.append(_rows(len(ticks), ticks, first_id=next_id))
        logical += [(m, t, next_id + i) for i, t in enumerate(ticks)]
        next_id += len(ticks)
    logical = logical[-20:]  # the ring's oldest row is match 0 tick 3: the rest was overwritten
    assert dataset.size == 20 and dataset._next == 3

    burn_in, length = 3, 5
    span = burn_in + length
    rng = np.random.default_rng(0)
    windows = dataset.sample_windows(400, length, burn_in, rng, max_gap=max_gap)
    assert windows["graphic"].shape == (400, span, 2, 2, 1) and windows["agent_states"].shape == (400, span, 10, 12)
    assert set(FIELDS) <= set(windows)
    ids = windows["agent_states"][:, :, 0, 0].astype(int) - 1  # row id, -1 = zeroed (invalid)
    by_id = {row_id: (m, t) for m, t, row_id in logical}
    position_of = {row_id: j for j, (_, _, row_id) in enumerate(logical)}
    seen_resets = set()
    for b in range(400):
        valid = windows["valid"][b]
        assert (ids[b][~valid] == -1).all() and (windows["labels"][b][~valid] == 0).all()
        assert (windows["tick"][b][~valid] == UNKNOWN_TICK).all()
        assert windows["loss_mask"][b].tolist() == (valid & (np.arange(span) >= burn_in)).tolist()
        assert windows["loss_mask"][b].any()
        # place the window from its first valid row (any row of the run determines the run)
        p = int(np.flatnonzero(valid)[0])
        expected_valid, expected_reset = _reference([(m, t) for m, t, _ in logical], position_of[ids[b, p]], p, span, max_gap)
        assert valid.tolist() == expected_valid.tolist()
        assert windows["reset"][b].tolist() == expected_reset.tolist()
        run = ids[b][valid]
        assert len({by_id[i][0] for i in run}) == 1  # never crosses a match
        assert np.array_equal(np.diff([position_of[i] for i in run]), np.ones(len(run) - 1))  # consecutive ring rows
        assert [by_id[i][1] for i in run] == windows["tick"][b][valid].tolist()
        assert (windows["match"][b][valid] == by_id[run[0]][0]).all()
        for q in np.flatnonzero(windows["reset"][b]):
            seen_resets.add(by_id[ids[b, q]])
    # resets: each match's first row in the ring (match 0's is tick 3: the overwrite boundary),
    # plus the tick-5 gap of match 2 unless max_gap bridges it
    expected = {(0, 3), (1, 0), (2, 0), (3, 0)} | ({(2, 5)} if max_gap < 3 else set())
    assert seen_resets == expected


def test_window_targets_cover_every_row_at_every_loss_position():
    dataset = DistillDataset("unused", capacity=50)
    dataset.append(_rows(10, range(10)))
    dataset.append(_rows(10, range(10), first_id=10))
    windows = dataset.sample_windows(20_000, length=4, burn_in=2, rng=np.random.default_rng(2))
    ids = windows["agent_states"][:, :, 0, 0].astype(int) - 1
    counts = np.bincount(ids[windows["loss_mask"]], minlength=20)
    # a row at a match's edge sits in fewer windows' loss part, but every row is trained
    assert (counts > 0).all()
    assert windows["reset"][:, :2].sum() > 0  # match starts inside burn-in happen too
