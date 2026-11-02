"""OOD-BP suite (blackout_env/train/distill/ood_bp.py): names are unique, groups expand, every descriptor builds."""

import pytest

from blackout_env.train.distill.ood_bp import GROUPS, OODBP, expand
from blackout_env.train.distill.opponents import check_opponent


def test_names_and_groups():
    names = [n for n, *_ in OODBP]
    assert len(names) == len(set(names)) == len(OODBP)
    assert expand("oodbp") == [d for _, d, _, _ in OODBP]
    for g in GROUPS:
        assert expand(f"oodbp:{g}") and all(d in expand("oodbp") for d in expand(f"oodbp:{g}"))
    assert expand("oodbp=exit_camp") == [dict((n, d) for n, d, _, _ in OODBP)["exit_camp"]]
    assert expand("B6") == ["B6"]
    with pytest.raises(ValueError):
        expand("oodbp:nope")


@pytest.mark.parametrize("name,descriptor", [(n, d) for n, d, _, _ in OODBP])
def test_descriptor_builds(name, descriptor):
    if descriptor.startswith("switch:"):  # v27_eval's SwitchPolicy splits it; build both halves
        _, _, rest = descriptor.split(":", 2)
        for half in rest.split("|"):
            check_opponent(half)
    else:
        check_opponent(descriptor)
