# "If we write if/else, the GPU slows down because warps move in lockstep."
# Goal: simulate the performance cost of branch divergence in a GPU warp.


def simulate_warp_execution(num_threads=32, has_branching=False):
    """Simulate how a single warp (32 threads) handles a branching instruction."""
    print(f"--- Warp Execution Simulation (Threads: {num_threads}) ---")

    if not has_branching:
        # All threads do the same math — maximum efficiency
        time_units = 1
        print("No branching: all threads executed in 1 cycle (lockstep).")
    else:
        # Branch divergence: threads split, paths execute serially
        # "One half must pause while the other half runs."
        time_units = 2
        print("Branch divergence detected: 'if' and 'else' paths executed serially.")
        print("Warp efficiency dropped by 50%.")

    return time_units


def check_gpu_warehouse():
    """
    "The GPU is mostly a giant, complex warehouse."
    Understanding chip area breakdown changes how you think about bottlenecks.
    """
    print("\n--- GPU Chip Composition ---")
    print("Memory (Registers / L1 / Shared SRAM): ~90% of chip area  (The Warehouse)")
    print("ALUs (Math units):                       ~10% of chip area  (The Workers)")
    print("\nMantra: Math is cheap, memory is expensive.")
    print("Bottleneck is almost always data movement, not computation.")


if __name__ == "__main__":
    simulate_warp_execution(has_branching=False)
    simulate_warp_execution(has_branching=True)
    print("-" * 30)
    check_gpu_warehouse()
