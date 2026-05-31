import torch

# "A diagram that explains how PyTorch works with memory."
# Transpose is just a view — it flips the strides, but doesn't copy data.
# This is fast, but leaves the tensor non-contiguous in memory.


x = torch.arange(16).view(4, 4)
print(f"Original shape:   {x.shape}  | strides: {x.stride()}  | contiguous: {x.is_contiguous()}")
# Strides [4, 1]: moving one row = jump 4 elements, moving one column = jump 1 element (row-major)

y = x.transpose(0, 1)
print(f"Transposed shape: {y.shape}  | strides: {y.stride()}  | contiguous: {y.is_contiguous()}")
# Strides [1, 4]: the stride order is reversed — memory is now non-contiguous (column-major in a row-major buffer)

# WHY this matters: some operations require contiguous memory and will crash otherwise.
# view() is the most common example — it only works on contiguous tensors.
try:
    _ = y.view(16)
except RuntimeError as e:
    print(f"\nview() on non-contiguous tensor: RuntimeError — {e}")

# The "Gold Standard" fix: .contiguous() allocates a fresh, row-major copy
if not y.is_contiguous():
    print("\nWarning: Data order in memory is messy. Fixing with .contiguous()...")
    y = y.contiguous()

print(f"\nAfter .contiguous(): strides: {y.stride()}  | contiguous: {y.is_contiguous()}")
# Strides are back to [4, 1] — clean row-major layout, view() works again

if __name__ == "__main__":
    # Visual: show that .contiguous() made a new copy (different data_ptr)
    x2 = torch.arange(4).view(2, 2)
    y2 = x2.transpose(0, 1)
    y2_c = y2.contiguous()
    print(f"\nSame storage as original? (transpose) : {x2.data_ptr() == y2.data_ptr()}")
    print(f"Same storage as original? (contiguous): {x2.data_ptr() == y2_c.data_ptr()}")
    # transpose shares storage; contiguous does not — that's the cost of the fix
