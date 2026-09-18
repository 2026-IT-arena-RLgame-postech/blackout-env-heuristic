"""DAgger relabelling in the trainer: labelled net-played rows get a set-valued BC term."""

import numpy as np
import torch

from blackout_env.train.qmix_trainer import QMIXConfig, QMIXTrainer
from blackout_env.train.replay_buffer import SOURCE_DATASET, SOURCE_SELF_VS_HEURISTIC


def _trainer(**overrides):
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


def test_labelled_net_rows_are_cloned_toward_the_label_set():
    trainer = _trainer()
    north, north_east = 2, 1  # label {N, NE}; the executed action (0 = east) is not in it
    _fill(trainer, demo=False, teach_bits=(1 << north) | (1 << north_east))
    loss, tensors, diag = _loss_and_diagnostics(trainer)
    assert diag["dagger_active_frac"] == 1.0 and diag["bc_active_frac"] == 1.0
    assert diag["dagger_raw"] is not None and diag["dagger_raw"] > 0

    # one gradient step on the BC-dominated loss raises the labelled directions' logits relative to the rest
    params = list(trainer.net.parameters())
    before = trainer.net(tensors["graphic"], tensors["team_state"], tensors["agent_states"], tau=None)[0][:, :5].detach()
    grads = torch.autograd.grad(loss.mean(), params, allow_unused=True)
    with torch.no_grad():
        for p, g in zip(params, grads):
            if g is not None:
                p -= 1e-2 * g
    after = trainer.net(tensors["graphic"], tensors["team_state"], tensors["agent_states"], tau=None)[0][:, :5].detach()
    def share(q):
        return torch.softmax(q, -1)[..., [north, north_east]].sum(-1).mean().item()
    assert share(after) > share(before)


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
