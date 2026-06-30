import torch
import torch.nn as nn
import torch.nn.functional as F

# Goal: understand why torch.compile is the 'easy speed' button.
# Two worlds of optimization:
#   Micro World: fusing GPU kernels to reduce HBM trips (this file)
#   Macro World: scaling out across multiple machines (see macro_scaling_ddp.py)


class NaiveBlock(nn.Module):
    """Each op launches a separate GPU kernel — separate HBM read/write per op."""
    def __init__(self, dim=256):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return torch.softmax(F.gelu(self.linear(x)), dim=-1)


# torch.compile fuses the ops into as few kernels as possible —
# data loaded from HBM once, all ops run in SRAM, written back once.
# "Use torch.compile for easy speed — it fuses kernels automatically."
# Requires PyTorch >= 2.0.


def simulate_scaling_out():
    print("--- Scaling Strategy ---")
    print("Micro World: optimize threads inside SRAM (kernel fusion via torch.compile).")
    print("Macro World: scale out across multiple machines (DDP / tensor parallelism).")
    print("Stack both to get the full picture.")


if __name__ == "__main__":
    simulate_scaling_out()

    dim   = 256
    batch = 32
    x     = torch.randn(batch, dim)

    naive_model = NaiveBlock(dim)

    try:
        compiled_model = torch.compile(naive_model)
        out = compiled_model(x)
        print(f"\ntorch.compile active — output shape: {out.shape}")
    except Exception as e:
        out = naive_model(x)
        print(f"\ntorch.compile unavailable ({e}) — running standard mode.")
        print(f"Output shape: {out.shape}")
