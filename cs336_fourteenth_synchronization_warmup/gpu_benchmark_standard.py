import torch
import time


# My Goal: Demonstrate the necessity of Warm-up and Synchronization.
# "Synchronize() is a stop sign for our measurement trap."

def complex_gpu_kernel():
    # Simulating a heavy GPU task
    x = torch.randn(5000, 5000, device='cuda')
    return x @ x


def professional_benchmark():
    # 1. THE WARM-UP (Wake up the GPU)
    print("Action: Starting Warm-up runs to initialize CUDA context...")
    for _ in range(5):
        _ = complex_gpu_kernel()
    torch.cuda.synchronize()  # Wait for warm-up to finish

    # 2. THE TIMED MEASUREMENT
    print("Action: Starting timed measurement with synchronization...")
    start_time = time.perf_counter()

    for _ in range(10):
        _ = complex_gpu_kernel()

    # "Without this stop sign, the timer is fake!"
    torch.cuda.synchronize()

    end_time = time.perf_counter()
    avg_time = (end_time - start_time) / 10
    print(f"Accurate Avg Time: {avg_time:.4f}s")


# --- THE DATA RATIO JOKE ---
def check_scaling_laws_history():
    # "Pre-2022, everyone followed Kaplan laws (Model size is king)."
    # "DeepMind (Chinchilla) realized data is more important."
    print("\n--- Scaling Law Evolution ---")
    print("2020 (Kaplan): Spend 100% on Model Size -> Result: Chunky/Hantal models.")
    print("2022 (Chinchilla): Split 50/50 between Model & Data -> Result: Compute-optimal.")


if torch.cuda.is_available():
    professional_benchmark()
    check_scaling_laws_history()
else:
    print("GPU not detected, but the logic is documented!")