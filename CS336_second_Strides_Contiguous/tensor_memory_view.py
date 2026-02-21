import torch

# "A diagram that explains how PyTorch works with memory."
# I learned that 'transpose' is just a view, it doesn't copy data.

x = torch.arange(16).view(4, 4)
print(f"Original Strides: {x.stride()}") # [4, 1]

# Transposing
y = x.transpose(0, 1)
print(f"Transposed Strides: {y.stride()}") # [1, 4] -> Memory is now non-contiguous!

# The "Gold Standard" fix from my notes:
if not y.is_contiguous():
    print("Warning: Data order in memory is messy. Fixing with .contiguous()...")
    y = y.contiguous() # I am making a clean copy in memory.