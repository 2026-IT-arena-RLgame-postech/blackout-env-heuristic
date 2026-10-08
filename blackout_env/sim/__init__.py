"""Fast Python/numba simulator of BlackOut that reproduces the Unity build (search-teacher S2).

See ``state.SimState`` for the API and docs/policy_v2_design_ko.md ("S2 시뮬레이터") for what is
modelled, how it was validated against recorded Unity traces, and the known divergences.
"""

from .state import DEFAULT_ORDER, SimState, obs_from_raw, raw_from_view

__all__ = ["SimState", "DEFAULT_ORDER", "obs_from_raw", "raw_from_view"]
