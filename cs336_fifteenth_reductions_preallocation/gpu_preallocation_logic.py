import torch

# My Goal: Understand the 'In-place' rule and Buffer Pre-allocation.
# "Memory allocation is slow; pre-allocate your warehouse space!"

def naive_allocation(x):
    # This creates new memory every time (Slow in a GPU kernel)
    return x + 1

def professional_preallocation(x, output_buffer):
    # "Instead of creating memory, we use the pre-allocated buffer."
    # In a real Triton/CUDA kernel, this would be an in-place operation
    torch.add(x, 1, out=output_buffer)
    return output_buffer

# --- THE SIMULATION ---
x = torch.randn(1024, device='cuda' if torch.cuda.is_available() else 'cpu')

# 1. Pre-allocate the buffer ONCE
output_buffer = torch.empty_like(x)

# 2. Use it in the loop
for _ in range(5):
    professional_preallocation(x, output_buffer)

print("Status: Operation completed using pre-allocated buffers.")
print("Engineering Rule: Out-of-place is slow, In-place is the Gold Standard.")