"""IQN distribution diagnostics: spread, skew, and the quantile-crossing health check."""

from __future__ import annotations

import torch

from blackout_env.train.qmix_trainer import QMIXTrainer

Q = 32


def _sorted_by_tau(values: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Scatter `values` (already in increasing order) to the positions tau puts them in."""
    out = torch.empty_like(values)
    out.scatter_(1, tau.argsort(dim=1), values)
    return out


def test_a_well_ordered_distribution_has_no_crossings():
    tau = torch.rand(4, Q)
    values = torch.linspace(-1.0, 1.0, Q).expand(4, Q).contiguous()
    stats = QMIXTrainer._quantile_diagnostics(_sorted_by_tau(values, tau), tau, torch.zeros(4, Q))
    assert stats["iqn_crossing_frac"] == 0.0
    assert stats["iqn_spread_p10_p90"] > 0


def test_crossings_are_counted():
    tau = torch.rand(4, Q)
    values = torch.linspace(1.0, -1.0, Q).expand(4, Q).contiguous()  # decreasing in tau: all cross
    stats = QMIXTrainer._quantile_diagnostics(_sorted_by_tau(values, tau), tau, torch.zeros(4, Q))
    assert stats["iqn_crossing_frac"] == 1.0


def test_a_collapsed_distribution_reads_as_zero_spread():
    """The failure mode where the head stops using tau and IQN degenerates to a scalar Q."""
    tau = torch.rand(4, Q)
    flat = torch.full((4, Q), 0.5)
    stats = QMIXTrainer._quantile_diagnostics(flat, tau, flat)
    assert stats["iqn_spread_p10_p90"] == 0.0
    assert stats["iqn_std"] == 0.0
    assert abs(stats["iqn_median"] - 0.5) < 1e-6


def test_skew_sign_follows_the_tail():
    tau = torch.rand(2, Q)
    upper_tail = torch.cat([torch.zeros(Q - 4), torch.linspace(1.0, 5.0, 4)]).expand(2, Q).contiguous()
    lower_tail = -upper_tail.flip(-1)
    up = QMIXTrainer._quantile_diagnostics(_sorted_by_tau(upper_tail, tau), tau, torch.zeros(2, Q))
    down = QMIXTrainer._quantile_diagnostics(_sorted_by_tau(lower_tail, tau), tau, torch.zeros(2, Q))
    assert up["iqn_skew"] > 0
    assert down["iqn_skew"] < 0


def test_target_spread_is_read_off_the_target_distribution():
    tau = torch.rand(3, Q)
    online = torch.linspace(-0.1, 0.1, Q).expand(3, Q).contiguous()
    target = torch.linspace(-5.0, 5.0, Q).expand(3, Q).contiguous()
    stats = QMIXTrainer._quantile_diagnostics(_sorted_by_tau(online, tau), tau, target)
    assert stats["iqn_target_spread"] > stats["iqn_spread_p10_p90"] * 10
