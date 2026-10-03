import sys, numpy as np, torch
from blackout_env.train.accel import pick_device, adamw, grad_scaler, to_device, pinned
from blackout_env.value.data import open_cache
from blackout_env.value.model import ValueNet, value_loss
from blackout_env.value.train import gather, predict_rows, compile_value_net, u8_graphic

mode = sys.argv[1]
device = pick_device("auto")
cache = open_cache("datasets/value_s1_sub4")
rows = np.flatnonzero(np.asarray(cache["split"]) == 0)
ev = np.sort(np.random.default_rng(5).choice(np.flatnonzero(np.asarray(cache["split"]) == 1), 20000, replace=False))
torch.manual_seed(0)
net = ValueNet().to(device)
net.load_trunk(torch.load("checkpoints/distill/frozen/p5a_v22.pt", map_location="cpu", weights_only=True)["policy_state"])
if "compile" in mode:
    compile_value_net(net, device)
opt = adamw([{"params": list(net.parameters())}], device, lr=3e-4, weight_decay=0.01)
scaler = grad_scaler(device, "fp16")
rng = np.random.default_rng(0)
def ev_nan(tag, amp):
    p = predict_rows(net, cache, ev, device, amp=amp)
    print(tag, "amp", amp, {k: int(np.isnan(v).sum()) for k, v in p.items()}, flush=True)
ev_nan("step0", torch.float16)
for step in range(int(sys.argv[2]) if len(sys.argv) > 2 else 300):
    b = to_device({k: pinned(v, device) for k, v in gather(cache, rows[rng.integers(0, len(rows), 256)]).items()}, device)
    with torch.autocast("cuda", dtype=torch.float16):
        out = net(u8_graphic(b["graphic"]), b["team_state"], b["agent_states"])
    loss, _ = value_loss(out, b, b["team_state"][:, 0] - b["team_state"][:, 1])
    opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); scaler.step(opt); scaler.update()
ev_nan("step300", torch.float16)
ev_nan("step300", None)
net.eval()
x = gather(cache, ev[:1024])
b = to_device(x, device)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    o = net._orig_mod(u8_graphic(b["graphic"]), b["team_state"], b["agent_states"]) if hasattr(net, "_orig_mod") else None
print("eager-forward-in-fp16 nan:", None if o is None else int(torch.isnan(o).sum()))
