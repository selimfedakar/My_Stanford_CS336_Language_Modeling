import torch
import time

"""
GOAL: Move from 'Waiting' (synchronize-then-compute) to 'Overlapping' (streams).
CONCEPT: Hiding latency — the 'macro' way to keep the GPU at 100% utilization.
"""


def stream_simulation():
    if not torch.cuda.is_available():
        print("CUDA not found. Imagine two highway lanes running in parallel!")
        return

    size = 4000
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')

    # 1. Create two independent 'highways'
    compute_stream  = torch.cuda.Stream()
    transfer_stream = torch.cuda.Stream()

    print("Starting overlap simulation: compute + secondary memory task in parallel...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # 2. THE OVERLAP TRICK
    # Stream A: heavy matrix computation
    with torch.cuda.stream(compute_stream):
        res = a @ b

    # Stream B: independent memory operation runs concurrently with Stream A
    # (In production: pair with actual H2D data transfer for maximum overlap)
    with torch.cuda.stream(transfer_stream):
        c = torch.randn(size, size, device='cuda')

    # 3. THE STOP SIGN: wait for both lanes to finish before proceeding
    torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    print(f"Parallel execution completed in {elapsed:.6f}s")
    print("Engineering rule: if your GPU is waiting for data, you are losing money.")


def data_is_the_new_oil():
    """
    "A small, well-fed model beats a starving giant."
    """
    print("\n--- The Chinchilla Optimality Note ---")
    print("Rule:   For every 2x increase in compute, you need 2x more data.")
    print("Proof:  Llama-3 8B trained on 15 trillion tokens — smaller model, stronger results.")
    print("Lesson: Don't just build a bigger engine; find a bigger ocean of fuel.")


if __name__ == "__main__":
    stream_simulation()
    data_is_the_new_oil()
