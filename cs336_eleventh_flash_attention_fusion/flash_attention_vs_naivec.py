# "I have been hammering on the memory bottleneck thing like a hammer hitting a nail."
# Goal: understand why kernel fusion (FlashAttention) beats the naive approach.


def naive_attention_workflow():
    """Naive attention: every intermediate result makes a round-trip to HBM."""
    print("--- Naive Approach (Slow) ---")
    print("1. Read Q, K from HBM")
    print("2. Compute scores → WRITE to HBM          (trip to warehouse)")
    print("3. Read scores from HBM → compute Softmax")
    print("4. WRITE Softmax result to HBM             (another trip)")
    print("5. Read Softmax from HBM → multiply with V")
    print("6. WRITE final result to HBM")
    print("Total HBM round-trips: many  (slow)")


def fused_attention_workflow():
    """FlashAttention: fuse all ops into one kernel, stay in SRAM."""
    print("\n--- Fused Approach (FlashAttention — Fast) ---")
    print("1. Load Q, K, V into fast SRAM once")
    print("2. Compute scores, Softmax, and V-weighted sum incrementally inside SRAM")
    print("   (never spill intermediates back to HBM)")
    print("3. WRITE only the final result to HBM")
    print("Total HBM round-trips: minimal  (extremely fast)")


def check_performance_logic():
    """
    The recomputation insight: it's faster to do math twice than to read memory.
    FlashAttention applies this during the backward pass — it recomputes attention
    from Q/K/V instead of storing the full attention matrix (O(N²) memory saved).
    """
    print("\n--- Hardware Intuition ---")
    print("Reading from HBM:  ~1000 cycles")
    print("Doing math (ALU):  ~1 cycle")
    print("Logic: 1000 extra ALU ops still cheaper than one HBM read.")
    print("       → Recompute during backward. Never store the N×N attention matrix.")


if __name__ == "__main__":
    naive_attention_workflow()
    fused_attention_workflow()
    check_performance_logic()
