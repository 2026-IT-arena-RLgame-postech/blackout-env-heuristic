"""Bisect the fp16 NaN of the whole-net compiled ValueNet (server scratch)."""
import sys, numpy as np, torch
from blackout_env.train.accel import pick_device, to_device
from blackout_env.value.data import open_cache
from blackout_env.value.model import ValueNet
from blackout_env.value.train import gather, u8_graphic

device = pick_device("auto")
cache = open_cache("datasets/value_s1_sub4")
state = torch.load("/home/cucumbersaurus/cuda_work/nan_state.pt", map_location="cpu")
bad = np.load("/home/cucumbersaurus/cuda_work/nan_rows.npy")
rows = np.sort(bad[:1024]) if len(bad) >= 1024 else np.sort(np.concatenate([bad, np.arange(1024 - len(bad))]))
x = to_device(gather(cache, rows), device)
g, ts, ag = u8_graphic(x["graphic"]), x["team_state"], x["agent_states"]
mode = sys.argv[1]


def run(net, grad=False):
    with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.float16):
        return net(g, ts, ag).float()


ref = ValueNet().to(device); ref.load_state_dict(state); ref.eval()
e = run(ref)
print("eager fp16 nonfinite", int((~torch.isfinite(e)).sum()))
net = ValueNet().to(device); net.load_state_dict(state); net.eval()
if mode == "whole":
    net.compile(dynamic=False)
elif mode == "trunk":
    net.trunk.graphic_encoder.compile(); net.trunk.vector_encoder.compile(); net.trunk.attention.compile()
elif mode == "fwd_fn":  # compile only the MyModel._trunk function + head as a function
    net.forward = torch.compile(net.forward, dynamic=False)
for i in range(4):
    c = run(net)
    print(mode, "call", i, "nonfinite", int((~torch.isfinite(c)).sum()), "max|c-e|", float((c - e)[torch.isfinite(c)].abs().max()))
# which part: compare the trunk tokens
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    te = ref.trunk._trunk(g, ts, ag).float()
    tc = torch.compile(net.trunk._trunk if mode != "whole" else ref.trunk._trunk, dynamic=False)(g, ts, ag).float()
print("trunk tokens: eager nonfinite", int((~torch.isfinite(te)).sum()), "compiled nonfinite", int((~torch.isfinite(tc)).sum()),
      "max|diff|", float((tc - te)[torch.isfinite(tc)].abs().max()), "max|te|", float(te.abs().max()))
