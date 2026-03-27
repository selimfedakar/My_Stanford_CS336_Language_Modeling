import torch
import time

"""
GOAL: Solve the 'Memory Bottleneck' identified in profiling.
CONCEPT: Arithmetic Intensity — maximizing math per byte of memory moved.
"""

@torch.compile # This is the 'Magic Button' that performs the fusion
def fused_operation(x, y):
    # Instead of 3 memory trips (Add -> Gelu -> Mul),
    # the compiler 'fuses' these into one GPU kernel.
    return (x + y).gelu() * 2

def naive_operation(x, y):
    # Each step here forces the GPU to write to HBM and read it back.
    # This is where the 'Memory Bottleneck' happens.
    a = x + y
    b = torch.nn.functional.gelu(a)
    return b * 2

def run_fusion_benchmark():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(10000, 10000, device=device)
    y = torch.randn(10000, 10000, device=device)

    # 1. Warm-up (Applying your 'Synchronization' rule)
    for _ in range(5):
        _ = naive_operation(x, y)
    if torch.cuda.is_available(): torch.cuda.synchronize()

    # 2. Benchmark Naive (Slow Memory Trips)
    start = time.perf_counter()
    for _ in range(20):
        _ = naive_operation(x, y)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    naive_time = (time.perf_counter() - start) / 20

    # 3. Benchmark Fused (Single Memory Trip)
    start = time.perf_counter()
    for _ in range(20):
        _ = fused_operation(x, y)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    fused_time = (time.perf_counter() - start) / 20

    print(f"--- Fusion Analysis ---")
    print(f"Naive Time: {naive_time:.6f}s")
    print(f"Fused Time: {fused_time:.6f}s")
    print(f"Speedup: {naive_time / fused_time:.2f}x")
    print("Engineering Rule: Don't move data if you don't have to.")

# --- THE CHINCHILLA VERDICT ---
def analyze_model_efficiency(name, params_b, tokens_t):
    """
    Refining the Scaling Law: Is the model 'Over-cooked' or 'Raw'?
    """
    ratio = (tokens_t * 1000) / params_b # Tokens in Billions
    print(f"\n--- Model Audit: {name} ---")
    if ratio > 200:
        print(f"Status: Over-trained / Inference-Optimal (Llama-3 Style)")
        print("Insight: We spent more on training to make the model smaller/faster for users.")
    elif ratio >= 20:
        print(f"Status: Chinchilla-Optimal.")
    else:
        print(f"Status: Under-trained (Kaplan Era).")

if __name__ == "__main__":
    run_fusion_benchmark()
    analyze_model_efficiency("Llama-3-8B", 8, 15) # 15 Trillion tokens