"""DAgger relabelling in the trainer: labelled net-played rows get a set-valued BC term."""

import numpy as np
import torch

from blackout_env.train.qmix_trainer import QMIXConfig, QMIXTrainer, set_margin_loss
from blackout_env.train.replay_buffer import SOURCE_DATASET, SOURCE_SELF_VS_HEURISTIC


def _trainer(**overrides):
    torch.manual_seed(0)
    np.random.seed(0)
    config = QMIXConfig(buffer_capacity=512, device="cpu", tb_log_dir=None, batch_size=8, spr_k=3,
                        bc_loss_alpha=1.0, **overrides)
    return QMIXTrainer(env=None, config=config)


def _fill(trainer, *, demo, teach_bits):
    rng = np.random.default_rng(0)
    buf = trainer.buffer_a
    shapes = (buf.graphic.shape[1:], buf.team_state.shape[1:], buf.agent_states.shape[1:])
    for i in range(120):
        states = rng.uniform(-1, 1, size=shapes[2]).astype(np.float32)
        states[:5, 2], states[5:, 2] = 1.0, -1.0  # team A view
        teach = np.zeros(10, dtype=np.uint8)
        teach[:5] = teach_bits
        buf.push(rng.random(shapes[0]).astype(np.float32), rng.random(shapes[1]).astype(np.float32),
                 states, np.zeros(10, dtype=np.int64), 0.0, bool(i % 40 == 39),
                 source=SOURCE_DATASET if demo else SOURCE_SELF_VS_HEURISTIC, demo=demo, teach=teach)


def _loss_and_diagnostics(trainer):
    batch = trainer._sample_batch(trainer.buffer_a, 8, n_step=3, gamma=0.99, beta=0.4)
    tensors = trainer._to_tensors(batch)
    loss, _ = trainer._forward_and_loss(tensors, collect_diagnostics=True)
    return loss, tensors, trainer._last_diagnostics


def test_labelled_net_rows_carry_a_bc_term():
    trainer = _trainer()
    north, north_east = 2, 1  # label {N, NE}; the executed action (0 = east) is not in it
    _fill(trainer, demo=False, teach_bits=(1 << north) | (1 << north_east))
    _, tensors, diag = _loss_and_diagnostics(trainer)
    assert diag["dagger_active_frac"] == 1.0 and diag["bc_active_frac"] == 1.0
    assert diag["dagger_raw"] is not None and diag["dagger_raw"] > 0

    # the relabelled term is in the loss: dropping the labels removes it
    with_labels = trainer._last_bc_loss
    tensors = dict(tensors)
    tensors["teach_bits"] = torch.zeros_like(tensors["teach_bits"])
    torch.manual_seed(0)
    trainer._forward_and_loss(tensors)
    assert with_labels > 0 and trainer._last_bc_loss == 0


def test_unlabelled_net_rows_and_demonstrations_are_unchanged():
    trainer = _trainer()
    _fill(trainer, demo=False, teach_bits=0)
    _, _, diag = _loss_and_diagnostics(trainer)
    assert diag["dagger_active_frac"] == 0.0 and diag["bc_active_frac"] == 0.0

    trainer = _trainer()
    _fill(trainer, demo=True, teach_bits=0b100)  # a label on a demonstration row is ignored
    _, _, diag = _loss_and_diagnostics(trainer)
    assert diag["dagger_active_frac"] == 0.0 and diag["bc_active_frac"] == 1.0


def test_dagger_weight_zero_turns_the_term_off():
    trainer = _trainer(dagger_weight=0.0)
    _fill(trainer, demo=False, teach_bits=0b100)
    loss, _, diag = _loss_and_diagnostics(trainer)
    assert diag["dagger_active_frac"] == 1.0  # labelled rows still reported...
    assert diag["bc_active_frac"] == 0.0      # ...but carry no BC weight


def test_set_margin_loss_is_zero_once_a_label_leads_by_the_margin():
    q = torch.tensor([[3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.8]])
    north_east, east, west = torch.zeros(1, 8), torch.zeros(1, 8), torch.zeros(1, 8)
    east[0, 0] = 1.0                       # labelled east leads by 0.2 < 0.5
    assert torch.isclose(set_margin_loss(q, east, 0.5), torch.tensor([0.3])).all()
    east[0, 7] = 1.0                       # {east, south-east}: best labelled 3.0 vs best other 1.0
    assert set_margin_loss(q, east, 0.5).item() == 0.0
    west[0, 4] = 1.0                       # label on a low action: must climb past the leader + margin
    assert torch.isclose(set_margin_loss(q, west, 0.5), torch.tensor([3.5])).all()
    assert set_margin_loss(q, torch.ones(1, 8), 0.5).item() == 0.0  # everything correct: nothing to beat


def test_margin_rows_get_a_gradient_only_while_the_margin_is_violated():
    trainer = _trainer(dagger_loss="margin", dagger_margin=0.5)
    _fill(trainer, demo=False, teach_bits=1 << 2)
    loss, _, diag = _loss_and_diagnostics(trainer)
    assert diag["dagger_active_frac"] == 1.0 and diag["dagger_raw"] > 0
    assert loss.mean().requires_grad


def test_rescale_is_run_11s_when_nothing_is_relabelled():
    trainer = _trainer(dagger_loss="margin")
    _fill(trainer, demo=True, teach_bits=0)
    batch = trainer._sample_batch(trainer.buffer_a, 8, n_step=3, gamma=0.99, beta=0.4)
    tensors = trainer._to_tensors(batch)
    raws = []
    for mode in ("margin", "ce"):
        trainer.cfg.dagger_loss = mode
        torch.manual_seed(3)  # same IQN quantile draws for both modes
        trainer._forward_and_loss(tensors, collect_diagnostics=True)
        raws.append(trainer._last_diagnostics["bc_raw"])
    assert abs(raws[0] - raws[1]) < 1e-6
