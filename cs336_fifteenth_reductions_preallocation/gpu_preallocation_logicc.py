import torch
import time

# "Memory allocation is slow; pre-allocate your warehouse space!"
# Goal: understand the 'out=' buffer pattern and its cost advantage over naive allocation.


def naive_allocation(x):
    # Creates a new tensor in memory every call — allocation cost every iteration
    return x + 1


def professional_preallocation(x, output_buffer):
    # "Instead of creating memory, we use the pre-allocated buffer."
    # torch.add with out= writes directly into the buffer — no allocation
    torch.add(x, 1, out=output_buffer)
    return output_buffer


def benchmark_both(size=1024, iterations=1000):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x             = torch.randn(size, device=device)
    output_buffer = torch.empty_like(x)

    # Warm-up
    for _ in range(10):
        naive_allocation(x)
        professional_preallocation(x, output_buffer)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Naive timing
    t0 = time.perf_counter()
    for _ in range(iterations):
        naive_allocation(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    naive_ms = (time.perf_counter() - t0) * 1000

    # Pre-allocated timing
    t0 = time.perf_counter()
    for _ in range(iterations):
        professional_preallocation(x, output_buffer)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prealloc_ms = (time.perf_counter() - t0) * 1000

    print(f"Naive allocation     ({iterations} iters): {naive_ms:.2f} ms")
    print(f"Pre-allocated buffer ({iterations} iters): {prealloc_ms:.2f} ms")
    if naive_ms > 0:
        print(f"Speedup: {naive_ms / prealloc_ms:.2f}x")
    print("Engineering rule: out-of-place = slow, pre-allocated buffer = Gold Standard.")


if __name__ == "__main__":
    benchmark_both()
