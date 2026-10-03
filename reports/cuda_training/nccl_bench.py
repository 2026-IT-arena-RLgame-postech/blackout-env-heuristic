import os, time, torch, torch.distributed as dist
r = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(r)
dist.init_process_group("nccl", device_id=torch.device("cuda", r))
for n in (1, 674336, 4 * 674336):
    x = torch.ones(n, device="cuda")
    for _ in range(10): dist.all_reduce(x)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(50): dist.all_reduce(x)
    torch.cuda.synchronize()
    if r == 0: print(f"world {dist.get_world_size()} allreduce {4*n/1e6:.2f} MB: {1000*(time.perf_counter()-t)/50:.3f} ms")
# a GPU-bound loop with no communication: does running N ranks slow each one?
a = torch.randn(4096, 4096, device="cuda", dtype=torch.half)
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(50): a @ a
torch.cuda.synchronize()
if r == 0: print(f"matmul 4096 fp16: {1000*(time.perf_counter()-t)/50:.2f} ms")
dist.destroy_process_group()
