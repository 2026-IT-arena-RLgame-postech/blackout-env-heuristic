import sys, numpy as np, torch, torch._dynamo
from blackout_env.train.accel import pick_device, to_device
from blackout_env.value.data import open_cache
from blackout_env.value.model import ValueNet
from blackout_env.value.train import gather, u8_graphic

device = pick_device("auto")
cache = open_cache("datasets/value_s1_sub4")
state = torch.load("/home/cucumbersaurus/cuda_work/nan_state.pt", map_location="cpu")
rows = np.sort(np.load("/home/cucumbersaurus/cuda_work/nan_rows.npy")[:1024])
x = to_device(gather(cache, rows), device)
g, ts, ag = u8_graphic(x["graphic"]), x["team_state"], x["agent_states"]
net = ValueNet().to(device); net.load_state_dict(state); net.eval()
m = net.trunk
for name, flag in (("default", None), ("no_mem_efficient", "mem"), ("math_only", "math")):
    torch._dynamo.reset()
    torch.backends.cuda.enable_mem_efficient_sdp(flag is None)
    torch.backends.cuda.enable_flash_sdp(flag is None)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        e = m._trunk(g, ts, ag).float()
        c = torch.compile(m._trunk, dynamic=False)(g, ts, ag).float()
    print(f"{name:18s} eager nonfinite {int((~torch.isfinite(e)).sum())} compiled nonfinite {int((~torch.isfinite(c)).sum())} "
          f"max|diff| {float((c - e).abs().max()):.4g}", flush=True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
# attention layers alone, fed the eager trunk input
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    import blackout_env.model.my_model as mm
    B = g.shape[0]
    from blackout_env.model.derived_obs import derived_map_channels, local_wall_features
    uf = local_wall_features(g, ag); g2 = torch.cat((g, derived_map_channels(g, ag)), 1); ag2 = torch.cat((ag, uf), -1)
    vis = m.graphic_encoder(g2); vec = m.vector_encoder(ag2, ts) + m.vec_slot_emb(m.vec_slot_ids)[None]
    spr = m.spr_cls_emb(m.spr_cls_ids)[None].expand(B, -1, -1)
    tok = torch.cat((vis, vec, spr), 1) + m.token_type_emb(m.token_type_ids)[None]
    rc, cc = m._token_grid_positions(ag2); rope = m.rope(rc, cc)
    torch._dynamo.reset()
    a_e = m.attention(tok, rope=rope).float()
    a_c = torch.compile(lambda t, r0, r1: m.attention(t, rope=(r0, r1)), dynamic=False)(tok, *rope).float()
    print("attention alone (rope as input): max|diff|", float((a_c - a_e).abs().max()), "nonfinite", int((~torch.isfinite(a_c)).sum()))
    print("tok dtype", tok.dtype, "max|tok|", float(tok.abs().max()), "rope dtype", rope[0].dtype)
