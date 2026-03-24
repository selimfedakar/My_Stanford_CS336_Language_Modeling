import torch
import time

"""
GOAL: Understand why 'Static Shapes' and 'KV Caching' are the keys to LLM speed.
CONCEPT: Moving from simple pre-allocation to Persistent State Buffers.
"""


class StaticKVCache:
    def __init__(self, max_batch_size, max_seq_len, head_dim):
        # We 'Claim' the memory upfront. The GPU never has to ask the OS for more.
        # This is the "Gold Standard" in Inference Engines (TensorRT / vLLM).
        self.cache = torch.zeros((max_batch_size, max_seq_len, head_dim),
                                 device='cuda' if torch.cuda.is_available() else 'cpu')
        print(f"--- Buffer Allocated: {self.cache.element_size() * self.cache.nelement() / 1024 ** 2:.2f} MB ---")

    def update_cache(self, new_token_embedding, index):
        """
        In-place update: No new memory is created.
        We simply overwrite the pre-allocated 'warehouse' slot.
        """
        self.cache[:, index:index + 1, :] = new_token_embedding
        return self.cache


def simulate_inference():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, T, D = 1, 1024, 512  # Batch, Seq_Len, Dim

    # 1. Initialize the Persistent World
    kv_engine = StaticKVCache(B, T, D)

    print("Starting token generation simulation...")
    start_time = time.time()

    for i in range(100):
        # Generate a 'new token'
        new_token = torch.randn(B, 1, D, device=device)

        # Update the static buffer IN-PLACE
        # Rule: If we used 'torch.cat' here, the GPU would lag due to re-allocation.
        kv_engine.update_cache(new_token, i)

    end_time = time.time()
    print(f"Status: 100 tokens processed in {end_time - start_time:.4f}s")
    print("Engineering Rule: Fragmentation is the enemy. Static buffers are the cure.")


if __name__ == "__main__":
    simulate_inference()