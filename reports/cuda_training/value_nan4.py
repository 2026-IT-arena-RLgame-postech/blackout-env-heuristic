"""Which piece of MyModel._trunk breaks under torch.compile (server scratch)."""
import numpy as np, torch
from blackout_env.train.accel import pick_device, to_device
from blackout_env.value.data import open_cache
from blackout_env.value.model import ValueNet
from blackout_env.value.train import gather, u8_graphic
from blackout_env.model.derived_obs import derived_map_channels, local_wall_features

device = pick_device("auto")
cache = open_cache("datasets/value_s1_sub4")
state = torch.load("/home/cucumbersaurus/cuda_work/nan_state.pt", map_location="cpu")
bad = np.load("/home/cucumbersaurus/cuda_work/nan_rows.npy")
rows = np.sort(bad[:1024])
x = to_device(gather(cache, rows), device)
g, ts, ag = u8_graphic(x["graphic"]), x["team_state"], x["agent_states"]
net = ValueNet().to(device); net.load_state_dict(state); net.eval()
m = net.trunk


def cmp(name, f, *a, amp=True):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        e = f(*a)
        c = torch.compile(f, dynamic=False)(*a)
    e, c = (e if isinstance(e, tuple) else (e,)), (c if isinstance(c, tuple) else (c,))
    errs = [float((ci.float() - ei.float()).abs().max()) for ei, ci in zip(e, c)]
    print(f"{name:28s} amp {amp}: max|diff| {errs}  max|e| {[float(ei.float().abs().max()) for ei in e]}", flush=True)


cmp("local_wall_features", local_wall_features, g, ag)
cmp("derived_map_channels", derived_map_channels, g, ag)
cmp("token_grid_positions", m._token_grid_positions, ag)
cmp("rope", lambda a: m.rope(*m._token_grid_positions(a)), ag)
cmp("_trunk fp32", m._trunk, g, ts, ag, amp=False)
cmp("_trunk fp16", m._trunk, g, ts, ag)


def pre(g, ts, ag):
    unit_features = local_wall_features(g, ag)
    g2 = torch.cat((g, derived_map_channels(g, ag)), dim=1)
    ag2 = torch.cat((ag, unit_features), dim=-1)
    return m.graphic_encoder(g2), m.vector_encoder(ag2, ts)


cmp("pre + encoders", pre, g, ts, ag)


def enc_only(g, ts, ag):
    unit_features = local_wall_features(g, ag)
    g2 = torch.cat((g, derived_map_channels(g, ag)), dim=1)
    return g2


cmp("graphic + derived", enc_only, g, ts, ag)
with torch.no_grad():
    unit_features = local_wall_features(g, ag)
    g2 = torch.cat((g, derived_map_channels(g, ag)), dim=1)
    ag2 = torch.cat((ag, unit_features), dim=-1)
cmp("graphic_encoder only", m.graphic_encoder, g2)
cmp("vector_encoder only", m.vector_encoder, ag2, ts)
