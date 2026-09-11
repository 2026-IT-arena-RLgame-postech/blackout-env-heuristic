"""Annealing schedules for BBF's n-step / gamma and PER's beta, as a function of training
progress (frac = env_step_count / anneal_steps, clamped to [0, 1])."""

import math


def linear_anneal(start: float, end: float, frac: float) -> float:
    frac = min(1.0, max(0.0, frac))
    return start + frac * (end - start)


def log_linear_anneal(start: float, end: float, frac: float) -> float:
    """Interpolates in log-space — BBF anneals gamma this way (e.g. 0.97 -> 0.997) rather
    than linearly, since discount factors compress meaningfully in log space."""
    frac = min(1.0, max(0.0, frac))
    return math.exp((1 - frac) * math.log(start) + frac * math.log(end))
