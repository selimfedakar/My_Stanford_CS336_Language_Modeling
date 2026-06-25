import time
import torch

# "Identifying the bottleneck: Is the GPU waiting for data from HBM?"
# Benchmarking = total wall-clock time. Profiling = where that time is spent.


def simulate_llm_step():
    time.sleep(0.01)                                      # Simulate HBM → SRAM transfer (slow)
    torch.randn(100, 100) @ torch.randn(100, 100)         # Compute (fast)


def benchmark_task(iterations=10):
    """Wall-clock measurement — tells you HOW slow, not WHY slow."""
    start = time.perf_counter()
    for _ in range(iterations):
        simulate_llm_step()
    total = time.perf_counter() - start
    print(f"Total wall-clock time ({iterations} steps): {total:.4f}s")


def profile_task():
    """
    "Profiling looks at where that time is being spent."
    Breaks the step into memory access vs compute.
    Note: For real GPU profiling use torch.profiler or CUDA events —
          time.perf_counter() is inaccurate for GPU ops without synchronize().
    """
    t0 = time.perf_counter()
    time.sleep(0.01)                                   # Memory trip (simulated)
    mem_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    torch.randn(100, 100) @ torch.randn(100, 100)      # ALU compute
    math_time = time.perf_counter() - t1

    print(f"\nProfiling report: Memory={mem_time:.4f}s | Math={math_time:.6f}s")
    print(f"Bottleneck: memory is {mem_time / math_time:.0f}x slower than compute.")


def check_training_status(model_name, params_b, tokens_b):
    """Chinchilla-optimal is ~20 tokens per parameter."""
    ratio = tokens_b / params_b
    status = "Compute-optimal" if ratio >= 20 else "Under-trained (data-starved)"
    print(f"\n{model_name}: {tokens_b}B tokens / {params_b}B params = {ratio:.0f} tok/param → {status}")


if __name__ == "__main__":
    benchmark_task()
    profile_task()
    print()
    check_training_status("GPT-3",    175, 300)    # 300B tokens — under-trained by Chinchilla
    check_training_status("Llama-2",    7, 2000)   # 2T tokens — Chinchilla-optimal
    check_training_status("Llama-3",    8, 15000)  # 15T tokens — over-trained for inference efficiency
