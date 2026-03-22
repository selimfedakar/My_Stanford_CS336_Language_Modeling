import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

"""
GOAL: Transition from single-kernel efficiency to massive multi-machine scaling.
CONCEPT: Combining torch.compile (Micro-optimization) with DDP (Macro-optimization).
Ahmet Selim FEDAKAR
"""


class ScalingModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Using GeLU as a nod to efficient kernel fusion
        self.net = nn.Sequential(
            nn.Linear(10, 128),
            nn.GELU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize the process group (The 'Macro' network)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_macro_simulation(rank, world_size):
    print(f"Running Macro-Scale Process on rank {rank}.")
    setup(rank, world_size)

    # 1. MICRO WORLD: Optimizing threads & kernels on the local device
    model = ScalingModel().to(rank)

    # Check if torch.compile is available (requires PyTorch 2.0+)
    try:
        model = torch.compile(model)
        print(f"Rank {rank}: Kernel fusion enabled via torch.compile")
    except Exception:
        print(f"Rank {rank}: Running in standard mode")

    # 2. MACRO WORLD: Scaling across multiple 'machines' (simulated by processes)
    ddp_model = DDP(model, device_ids=None)  # device_ids=None for CPU simulation

    # Mock Data
    data = torch.randn(20, 10)
    labels = torch.randn(20, 10)

    # Training Step
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.001)
    optimizer.zero_grad()

    outputs = ddp_model(data)
    loss = torch.nn.functional.mse_loss(outputs, labels)
    loss.backward()  # Gradients are synchronized across the 'Macro' cluster here
    optimizer.step()

    if rank == 0:
        print("\n--- Scaling Strategy Summary ---")
        print("Success: Synchronized gradients across the Macro World.")
        print("Optimization: Micro-level kernels fustigated via torch.compile.")

    cleanup()


if __name__ == "__main__":
    # We simulate 2 separate 'machines' or GPUs on your local system
    world_size = 2
    mp.spawn(run_macro_simulation,
             args=(world_size,),
             nprocs=world_size,
             join=True)