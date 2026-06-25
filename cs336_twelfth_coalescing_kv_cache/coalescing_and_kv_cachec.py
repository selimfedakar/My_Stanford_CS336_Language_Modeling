import torch

# "Think of it like a grocery store: buy the whole bag, not one item."
# Goal: understand memory coalescing and why KV cache is essential for inference.


def simulate_kv_cache(sequence_length, hidden_dim):
    """
    KV cache: store computed keys and values so we never recompute the past.
    Without it: every new token requires reprocessing the full context — O(N²).
    With it: only the new token's K and V are computed — O(N) total.
    """
    print(f"--- Inference with KV Cache (seq_len={sequence_length}, hidden={hidden_dim}) ---")

    cache_k = torch.zeros(1, sequence_length, hidden_dim)
    cache_v = torch.zeros(1, sequence_length, hidden_dim)

    # Simulate generating tokens one by one
    for t in range(min(sequence_length, 3)):  # Show first 3 steps for clarity
        new_k = torch.randn(1, 1, hidden_dim)
        new_v = torch.randn(1, 1, hidden_dim)
        cache_k[0, t] = new_k[0, 0]
        cache_v[0, t] = new_v[0, 0]
        print(f"  Step {t + 1}: stored new K/V | cache filled: {t + 1}/{sequence_length} slots")

    print("  'We load model weights once and use stored past keys/values.'")
    print(f"  Cache memory: {2 * cache_k.nelement() * 4 / 1024:.1f} KB (both K and V, FP32)")


def check_coalescing_efficiency(is_row_major=True):
    """
    Coalesced access: all 32 warp threads read from adjacent memory addresses.
    "Row-reading is faster because threads grab numbers from the same bag."
    """
    if is_row_major:
        print("\nAccess: Coalesced (row-wise)")
        print("Efficiency: 100% — 1 cache-line fetch, all 128 bytes used")
    else:
        print("\nAccess: Strided (column-wise)")
        print("Efficiency: Low — 1 useful element per 128-byte fetch (cache-line waste)")


if __name__ == "__main__":
    simulate_kv_cache(sequence_length=10, hidden_dim=512)
    check_coalescing_efficiency(is_row_major=True)
    check_coalescing_efficiency(is_row_major=False)
