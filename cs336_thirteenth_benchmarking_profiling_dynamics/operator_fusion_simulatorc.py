import torch
import torch.nn.functional as F
import time

"""
GOAL: Solve the 'Memory Bottleneck' identified in profiling.
CONCEPT: Arithmetic Intensity — maximizing math per byte of memory moved.
Requires PyTorch >= 2.0 for torch.compile.
"""


@torch.compile  # The 'Magic Button': fuses the three ops into one GPU kernel
def fused_operation(x, y):
    # Compiler sees: add → gelu → mul — and merges them into one kernel
    # Result: data loaded from HBM once, all three ops run in SRAM, written back once
    return (x + y).gelu() * 2


def naive_operation(x, y):
    # Each op forces a write-to-HBM / read-from-HBM cycle — this is the bottleneck
    a = x + y
    b = F.gelu(a)
    return b * 2


def run_fusion_benchmark(iterations=20):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(10_000, 10_000, device=device)
    y = torch.randn(10_000, 10_000, device=device)

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Warm-up — essential before any benchmark to avoid cold-start bias
    for _ in range(5):
        naive_operation(x, y)
    sync()

    # Naive timing
    t0 = time.perf_counter()
    for _ in range(iterations):
        naive_operation(x, y)
    sync()
    naive_time = (time.perf_counter() - t0) / iterations

    # Fused timing
    t0 = time.perf_counter()
    for _ in range(iterations):
        fused_operation(x, y)
    sync()
    fused_time = (time.perf_counter() - t0) / iterations

    print("--- Operator Fusion Analysis ---")
    print(f"Naive time (avg): {naive_time * 1000:.3f} ms")
    print(f"Fused time (avg): {fused_time * 1000:.3f} ms")
    print(f"Speedup:          {naive_time / fused_time:.2f}x")
    print("Engineering rule: Don't move data if you don't have to.")


def analyze_model_efficiency(name, params_b, tokens_b):
    """
    Is the model 'over-cooked' (inference-optimal) or 'raw' (under-trained)?
    Tokens in billions; ratio = tokens per parameter.
    """
    ratio = tokens_b / params_b
    if ratio > 200:
        status = "Over-trained / inference-optimal (Llama-3 style)"
        note   = "More training cost → smaller/faster model for users at inference time."
    elif ratio >= 20:
        status = "Chinchilla-optimal"
        note   = "Equal scaling of model size and data."
    else:
        status = "Under-trained (Kaplan era)"
        note   = "Model capacity wasted — more data would improve loss."

    print(f"\n--- Model Audit: {name} ---")
    print(f"  {tokens_b}B tokens / {params_b}B params = {ratio:.0f} tok/param")
    print(f"  Status: {status}")
    print(f"  Insight: {note}")


if __name__ == "__main__":
    run_fusion_benchmark()
    analyze_model_efficiency("Llama-3-8B",  8,  15_000)  # 15T tokens
    analyze_model_efficiency("Chinchilla",  70, 1_400)   # ~20 tok/param
    analyze_model_efficiency("GPT-3",       175, 300)    # Under-trained by modern standards
