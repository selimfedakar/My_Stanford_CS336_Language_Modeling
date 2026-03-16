import time
import torch


# My Goal: Understand the difference between total time and step-by-step profiling.
# "Identifying the bottleneck: Is the GPU waiting for data from HBM?"

def simulate_llm_step():
    # 1. Memory Access (HBM -> SRAM) - Slow part
    time.sleep(0.01)
    # 2. Compute (ALU Math) - Fast part
    torch.randn(100, 100) @ torch.randn(100, 100)


# --- BENCHMARKING (Wall-clock) ---
def benchmark_task(iterations=10):
    start = time.perf_counter()
    for _ in range(iterations):
        simulate_llm_step()
    total_time = time.perf_counter() - start
    print(f"Total Wall-clock Time: {total_time:.4f}s (End-to-end measurement)")


# --- PROFILING (Drill Down) ---
def profile_task():
    # "Profiling look at where that time is being spent."
    start_mem = time.perf_counter()
    time.sleep(0.01)  # Simulating memory trip
    mem_time = time.perf_counter() - start_mem

    start_math = time.perf_counter()
    torch.randn(100, 100) @ torch.randn(100, 100)
    math_time = time.perf_counter() - start_math

    print(f"Profiling Report: Memory={mem_time:.4f}s, Math={math_time:.4f}s")
    print(f"Bottleneck Identified: Memory is {mem_time / math_time:.1f}x slower than Math.")


# --- STRATEGIC DATA RATIO ---
def check_training_status(params_b, tokens_b):
    # Chinchilla-optimal is roughly 20 tokens per parameter
    ratio = tokens_b / params_b
    if ratio < 20:
        print(f"Ratio: {ratio:.1f} -> Status: Under-trained (Data-starved!)")
    else:
        print(f"Ratio: {ratio:.1f} -> Status: Compute-optimal Training.")


benchmark_task()
profile_task()
check_training_status(7, 100)  # Llama-2 style: 7B params with 100B tokens