import torch

from blackout_env.official.obs_mask import BATTERY_CONST, mask_canonical


def test_mask_oracle_keeps_battery_amounts():
    g = torch.zeros(1, 13, 24, 24)
    g[0, 8, 3, 3] = 2 / 3
    a = torch.zeros(1, 10, 12)
    a[0, 7, 4] = 1.0
    t = torch.zeros(1, 4)
    g3, _, a3 = mask_canonical(g, t, a, 3)
    g4, _, a4 = mask_canonical(g, t, a, 4)
    assert abs(float(g3[0, 8, 3, 3]) - BATTERY_CONST) < 1e-6 and abs(float(a3[0, 7, 4]) - BATTERY_CONST) < 1e-6
    assert abs(float(g4[0, 8, 3, 3]) - 2 / 3) < 1e-6 and float(a4[0, 7, 4]) == 1.0
