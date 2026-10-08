import sys, torch
sys.path.insert(0, "tests")
import test_accel as T
from blackout_env.model.my_model import MyModel
torch.backends.cudnn.deterministic = "--det" in sys.argv
dev = torch.device("cuda")
torch.manual_seed(0)
net = MyModel(n_actions=32, policy_head="logits", angle_head=True, angle_arc_bins=1).to(dev).train()
g, t, a = T._mymodel_batch(64, dev)
def loss(m):
    out = m(g, t, a, q_units=5, angle=True)
    return out[0].float().square().mean() + out[5].float().square().mean() + out[3].float().square().mean()
e1 = T._grads(net, lambda: loss(net))
e2 = T._grads(net, lambda: loss(net))
c = T._grads(net, lambda: loss(torch.compile(net)))
# fp64 eager reference
n64 = MyModel(n_actions=32, policy_head="logits", angle_head=True, angle_arc_bins=1).to(dev).train()
n64.load_state_dict(net.state_dict()); n64 = n64.double()
g, t, a = g.double(), t.double(), a.double()
r = T._grads(n64, lambda: loss(n64))
worst = []
for k in e1:
    rel = lambda x: float((x.double() - r[k]).norm() / r[k].norm().clamp_min(1e-30))
    worst.append((rel(c[k]), rel(e1[k]), float((e1[k] - e2[k]).norm() / e1[k].norm().clamp_min(1e-30)), k))
worst.sort(reverse=True)
print("rel-to-fp64:  compiled  eager  | eager run-to-run")
for w in worst[:12]:
    print(f"{w[0]:.2e} {w[1]:.2e} | {w[2]:.2e} {w[3]}")
for k in ("token_type_emb.weight", "spr_cls_emb.weight", "vec_slot_emb.weight"):
    print(k, f"compiled/eager norm ratio {float(c[k].norm() / e1[k].norm()):.6f}")
