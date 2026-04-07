import torch

# My Goal: Understand why Kernel Fusion is faster than the Naive approach.
# "I have been hammering on the memory bottleneck thing like a hammer hitting a nail."

def naive_attention_workflow():
    """
    Simulating the multiple trips to HBM (Main Memory).
    """
    print("--- Naive Approach (Slow) ---")
    print("1. Read Q, K from HBM")
    print("2. Compute Scores and WRITE result to HBM") # Trip to warehouse
    print("3. Read Scores from HBM to calculate Softmax")
    print("4. WRITE Softmax result to HBM") # Another trip
    print("5. Read Softmax from HBM to multiply with V")
    print("6. WRITE Final result to HBM")
    print("Total HBM Trips: High (Slow)")

def fused_attention_workflow():
    """
    Simulating Kernel Fusion (FlashAttention).
    """
    print("\n--- Fused Approach (FlashAttention - Fast) ---")
    print("1. Load Q, K, V into fast SRAM ONCE") # One trip!
    print("2. Compute Scores, Softmax, and Multi-V incrementally inside SRAM")
    print("   (Avoiding HBM until the very end)")
    print("3. WRITE ONLY the final result back to HBM")
    print("Total HBM Trips: Minimal (Extremely Fast)")

# --- THE RECOMPUTATION SECRET ---
# "It's faster to do math twice than to read memory."
def check_performance_logic():
    print("\n--- Hardware Intuition ---")
    print("Reading from HBM: ~1000 cycles")
    print("Doing Math (ALU): ~1 cycle")
    print("Logic: Doing 100 extra math ops is still faster than 1 memory read.")

# Running the simulation
naive_attention_workflow()
fused_attention_workflow()
check_performance_logic()