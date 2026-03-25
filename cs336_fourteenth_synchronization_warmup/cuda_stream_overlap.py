import torch
import time

"""
GOAL: Move from 'Waiting' (Synchronize) to 'Overlapping' (Streams).
CONCEPT: Hiding Latency—The "Macro" way to keep the GPU at 100% utilization.
"""


def stream_simulation():
    if not torch.cuda.is_available():
        print("CUDA not found. Imagine two highway lanes running in parallel!")
        return

    # 1. Create two independent 'Highways'
    compute_stream = torch.cuda.Stream()
    transfer_stream = torch.cuda.Stream()

    # Pre-allocate (Building on your previous 'Warehouse' rule)
    size = 4000
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')

    print("Action: Starting Overlap Simulation (Compute + Data Transfer)")

    # Synchronize before starting the benchmark
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # 2. THE OVERLAP TRICK
    with torch.cuda.stream(compute_stream):
        # Heavy computation on Stream A
        res = a @ b

    with torch.cuda.stream(transfer_stream):
        # While Stream A is busy, Stream B handles a different memory task
        # This 'hides' the time it takes to move data
        c = torch.randn(size, size, device='cuda')
        c = c + 1

    # 3. THE STOP SIGN: Wait for both lanes to merge
    torch.cuda.synchronize()

    end_time = time.perf_counter()
    print(f"Status: Parallel execution completed in {end_time - start_time:.6f}s")
    print("Engineering Rule: If your GPU is waiting for data, you are losing money.")


# --- THE CHINCHILLA ADDENDUM ---
def data_is_the_new_oil():
    """
    "A small, well-fed model beats a starving giant."
    """
    print("\n--- The Chinchilla Optimality Note ---")
    print("Rule: For every 2x increase in compute, you need 2x more data.")
    print("Result: Llama-3 (8B) trained on 15 Trillion tokens is the proof.")
    print("Lesson: Don't just build a bigger engine; find a bigger ocean of fuel.")


if __name__ == "__main__":
    stream_simulation()
    data_is_the_new_oil()