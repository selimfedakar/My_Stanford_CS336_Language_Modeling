import torch
import time

# "Synchronize() is a stop sign for our measurement trap."
# Goal: demonstrate why warm-up and synchronization are non-negotiable in GPU benchmarking.


def complex_gpu_kernel():
    x = torch.randn(5000, 5000, device='cuda')
    return x @ x


def professional_benchmark(warmup=5, iterations=10):
    # 1. WARM-UP — initializes CUDA context, JIT kernels, driver state
    # Without this, the first few iterations pay a one-time setup cost
    print(f"Starting {warmup} warm-up runs...")
    for _ in range(warmup):
        complex_gpu_kernel()
    torch.cuda.synchronize()  # Drain the GPU queue before measuring

    # 2. TIMED MEASUREMENT
    print(f"Starting {iterations} timed iterations...")
    t0 = time.perf_counter()
    for _ in range(iterations):
        complex_gpu_kernel()

    # "Without this stop sign, the timer is fake!"
    # GPU ops are asynchronous — perf_counter() returns immediately while GPU is still running
    torch.cuda.synchronize()

    avg_ms = (time.perf_counter() - t0) / iterations * 1000
    print(f"Accurate avg time: {avg_ms:.2f} ms")


def check_scaling_laws_history():
    print("\n--- Scaling Law Evolution ---")
    print("2020 (Kaplan):     Spend budget on model size        → over-parameterized, under-trained")
    print("2022 (Chinchilla): Split budget between model & data → compute-optimal")
    print("2023+ (Llama-3):   Over-train small models on massive data → inference-optimal")


if __name__ == "__main__":
    if torch.cuda.is_available():
        professional_benchmark()
    else:
        print("GPU not detected — synchronization logic documented above is still valid.")

    check_scaling_laws_history()
