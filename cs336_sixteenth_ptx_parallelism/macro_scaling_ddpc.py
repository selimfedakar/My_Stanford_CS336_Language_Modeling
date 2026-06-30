import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

"""
GOAL: Transition from single-kernel efficiency to massive multi-machine scaling.
CONCEPT: torch.compile (Micro) + DDP (Macro) — the full optimization stack.
"""


class ScalingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 128),
            nn.GELU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # 'gloo' backend works on CPU — use 'nccl' for multi-GPU
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_macro_simulation(rank, world_size):
    print(f"Macro-scale process started — rank {rank}/{world_size}.")
    setup(rank, world_size)

    # MICRO WORLD: kernel fusion on the local device
    model = ScalingModel()  # CPU for portability (swap .to(f'cuda:{rank}') for multi-GPU)
    try:
        model = torch.compile(model)
        if rank == 0:
            print("Kernel fusion enabled via torch.compile (PyTorch >= 2.0)")
    except Exception:
        if rank == 0:
            print("torch.compile unavailable — running in standard mode")

    # MACRO WORLD: gradient synchronization across processes
    # DDP wraps the model; .backward() all-reduces gradients automatically
    ddp_model = DDP(model, device_ids=None)  # device_ids=None = CPU mode

    data   = torch.randn(20, 10)
    labels = torch.randn(20, 10)

    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss = nn.functional.mse_loss(ddp_model(data), labels)
    loss.backward()   # Gradients synchronized across the macro cluster here
    optimizer.step()

    if rank == 0:
        print("\n--- Scaling Strategy Summary ---")
        print("Micro: kernel fusion via torch.compile — fewer HBM trips per device.")
        print("Macro: DDP gradient sync — all-reduce across the cluster.")
        print(f"Combined loss: {loss.item():.4f}")

    cleanup()


if __name__ == "__main__":
    world_size = 2  # Simulates 2 machines / GPUs on local system
    mp.spawn(run_macro_simulation, args=(world_size,), nprocs=world_size, join=True)
