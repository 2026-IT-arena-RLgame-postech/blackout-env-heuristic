import math

import pytest

from blackout_env.train.distill.run import Learner, lr_at, lr_is_scheduled, parse_args


def _args(*extra):
    return parse_args(["--build", "x", "--device", "cpu", "--steps", "1000", "--lr", "3e-4", *extra])


def test_constant_is_the_default_and_never_touches_the_optimizer():
    args = _args()
    assert (args.lr_schedule, args.lr_warmup, args.lr_final) == ("constant", 0, 0.1)
    assert not lr_is_scheduled(args)
    assert all(lr_at(s, args) == 3e-4 for s in (0, 1, 500, 999, 5000))
    learner = Learner(args)
    for group in learner.optimizer.param_groups:
        group["lr"] = 1.234e-4  # e.g. restored from a checkpoint: the default keeps it, as before
    learner.set_lr(999)
    assert all(group["lr"] == 1.234e-4 for group in learner.optimizer.param_groups)


@pytest.mark.parametrize("schedule", ["cosine", "linear"])
def test_decay_values_without_warmup(schedule):
    args = _args("--lr-schedule", schedule, "--lr-final", "0.1")
    assert lr_at(0, args) == pytest.approx(3e-4)
    assert lr_at(999, args) == pytest.approx(3e-5)  # the last step (--steps - 1) = lr x final
    assert lr_at(5000, args) == pytest.approx(3e-5)  # held after the end
    mid = 999 / 2
    expected_shape = 0.5 if schedule == "linear" else 0.5 * (1 + math.cos(math.pi * 0.5))
    assert lr_at(mid, args) == pytest.approx(3e-4 * (0.1 + 0.9 * expected_shape))
    values = [lr_at(s, args) for s in range(1000)]
    assert all(a >= b for a, b in zip(values, values[1:]))  # monotone


@pytest.mark.parametrize("schedule", ["constant", "cosine", "linear"])
def test_warmup_rises_linearly_from_zero_then_follows_the_schedule(schedule):
    args = _args("--lr-schedule", schedule, "--lr-warmup", "100", "--lr-final", "0.2")
    assert lr_is_scheduled(args)
    assert lr_at(0, args) == pytest.approx(3e-4 / 100)
    assert lr_at(49, args) == pytest.approx(3e-4 * 0.5)
    assert lr_at(99, args) == pytest.approx(3e-4)  # warm-up end
    assert lr_at(100, args) == pytest.approx(3e-4)  # decay starts at the peak
    # decay midpoint: halfway between the warm-up end and the last step
    mid = 100 + (999 - 100) / 2
    shape = {"constant": 1.0, "linear": 0.5, "cosine": 0.5}[schedule]
    expected = 3e-4 if schedule == "constant" else 3e-4 * (0.2 + 0.8 * shape)
    assert lr_at(mid, args) == pytest.approx(expected)
    assert lr_at(999, args) == pytest.approx(3e-4 if schedule == "constant" else 3e-4 * 0.2)


def test_resume_continues_the_same_curve_and_a_longer_run_recomputes_it(tmp_path):
    args = _args("--lr-schedule", "cosine", "--lr-warmup", "10", "--steps", "40")

    def lrs(learner, steps):
        out = []
        for step in steps:
            learner.set_lr(step)
            out.append(learner.optimizer.param_groups[0]["lr"])
        return out

    straight = lrs(Learner(args), range(40))
    first = Learner(args)
    head = lrs(first, range(25))
    resumed = Learner(args)  # a restart: the optimizer state (with the last lr) is loaded, then the curve continues
    resumed.optimizer.load_state_dict(first.optimizer.state_dict())
    assert head + lrs(resumed, range(25, 40)) == straight

    longer = _args("--lr-schedule", "cosine", "--lr-warmup", "10", "--steps", "80")
    assert lr_at(25, longer) > lr_at(25, args)  # --steps extended on resume: the curve is recomputed from the step
    assert lr_at(79, longer) == pytest.approx(lr_at(39, args))
