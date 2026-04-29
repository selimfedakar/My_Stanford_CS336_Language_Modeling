# My Goal: Simulate the performance cost of Branch Divergence in a GPU Warp.
# "If we write if/else, the GPU slows down because warps move in lockstep."

def simulate_warp_execution(num_threads=32, has_branching=False):
    """
    Simulating how a single Warp (32 threads) processes instructions.
    """
    print(f"--- Warp Execution Simulation (Threads: {num_threads}) ---")

    if not has_branching:
        # All threads do the same math (High efficiency)
        time_units = 1
        print("✅ No Branching: All threads executed in 1 cycle (Lockstep).")
    else:
        # Threads are split (Branch Divergence)
        # "One half must pause while the other half runs."
        time_units = 2
        print("⚠️ Branch Divergence Detected: The 'if' and 'else' paths executed serially.")
        print("Status: Warp efficiency dropped by 50%.")

    return time_units


# --- HARDWARE REALITY CHECK ---
def check_gpu_warehouse():
    # "The GPU is mostly a giant, complex warehouse."
    print("\n--- GPU Chip Composition ---")
    print("Memory (Registers/L1/Shared): 90% of chip area (The Warehouse)")
    print("ALUs (Math Units): 10% of chip area (The Workers)")
    print("Mantra: Math is cheap, memory is expensive.")


# Running the simulation
simulate_warp_execution(has_branching=False)
simulate_warp_execution(has_branching=True)
print("-" * 30)
check_gpu_warehouse()