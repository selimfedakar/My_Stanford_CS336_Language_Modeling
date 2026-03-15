import torch


# My Goal: Understand the speed difference between Coalesced and Strided access.
# "Think of it like a grocery store: buy the whole bag, not one item."

def simulate_kv_cache(sequence_length, hidden_dim):
    """
    Simulating how KV Cache saves us from recomputing the past.
    """
    print(f"--- Inference with KV Cache (Seq Length: {sequence_length}) ---")

    # 1. Without Cache: We process EVERYTHING for every new token (Slow)
    # Total ops = 1 + 2 + 3 + ... + N = O(N^2)

    # 2. With Cache: We only compute the NEW token
    # "We load model weights once and use stored past keys/values."
    new_k = torch.randn(1, 1, hidden_dim)
    new_v = torch.randn(1, 1, hidden_dim)

    print("Action: Storing new K and V into the 'Warehouse' (Cache).")
    print("Status: Only computing the current token. Historical context is preserved.")


# --- HARDWARE REALITY ---
def check_coalescing_efficiency(is_row_major=True):
    # "Row-reading is faster because threads grab numbers from the same bag."
    if is_row_major:
        print("\n✅ Access: Coalesced (Row-wise)")
        print("Efficiency: 100% (1 Trip = 1 Bag of 128 bytes used fully)")
    else:
        print("\n🚨 Access: Strided/Column-wise")
        print("Efficiency: Low (Many trips for mostly empty bags)")


# Running the simulation
simulate_kv_cache(10, 512)
check_coalescing_efficiency(is_row_major=True)
check_coalescing_efficiency(is_row_major=False)