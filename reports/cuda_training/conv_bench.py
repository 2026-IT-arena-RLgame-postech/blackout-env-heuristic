import time, torch, sys
from torch import nn
torch.backends.cudnn.benchmark = True
dev = "cuda"
B = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
specs = [l.split(",") for l in sys.argv[2].split(";")] if len(sys.argv) > 2 else [["19", "32", "1", "24"], ["32", "32", "2", "24"], ["32", "64", "2", "12"]]
for cl in (0, 1):
    for dt in (torch.float32, torch.float16):
        tot = 0
        for cin, cout, stride, hw in specs:
            cin, cout, stride, hw = int(cin), int(cout), int(stride), int(hw)
            conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1).to(dev, dt)
            x = torch.randn(B, cin, hw, hw, device=dev, dtype=dt, requires_grad=(cin != 19 and cin != 24))
            if cl:
                conv = conv.to(memory_format=torch.channels_last)
                x = x.detach().to(memory_format=torch.channels_last).requires_grad_(x.requires_grad)
            for i in range(10):
                conv(x).float().sum().backward()
            torch.cuda.synchronize(); t = time.perf_counter()
            for i in range(50):
                conv(x).float().sum().backward()
            torch.cuda.synchronize(); ms = 1000 * (time.perf_counter() - t) / 50
            tot += ms
            print(f"cl {cl} {dt} conv {cin}->{cout} s{stride} {hw}x{hw}: {ms:.3f} ms fwd+bwd")
        print(f"   total {tot:.3f}")
