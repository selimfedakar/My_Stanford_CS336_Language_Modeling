import torch

# My Goal: Understand why torch.compile is the 'easy speed' button.

def naive_ops(x):
    # Launches separate kernels for each op (Slow HBM trips)
    return torch.softmax(torch.nn.functional.gelu(torch.nn.Linear(10, 10)(x)), dim=-1)

# "Use torch.compile for cosy speed - it fustigates kernels automatically."
# (Note: Requires a compatible environment)
# optimized_model = torch.compile(naive_ops)

def simulate_scaling_out():
    """
    "A single GPU isn't enough; we need multi-machine parallelism."
    """
    print("--- Scaling Strategy ---")
    print("Micro World: Optimizing threads inside SRAM (Kernel Fusion).")
    print("Macro World: Scaling out across multiple machines (Distributed Training).")

simulate_scaling_out()