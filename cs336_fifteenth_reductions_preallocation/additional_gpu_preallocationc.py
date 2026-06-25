import torch
import time

"""
GOAL: Understand why 'Static Shapes' and 'KV Caching' are the keys to LLM inference speed.
CONCEPT: Moving from simple pre-allocation to Persistent State Buffers.
"""


class StaticKVCache:
    def __init__(self, max_batch_size, max_seq_len, head_dim):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # "We 'claim' the memory upfront. The GPU never has to ask the OS for more."
        # This is the Gold Standard in inference engines (TensorRT, vLLM).
        self.cache = torch.zeros(
            (max_batch_size, max_seq_len, head_dim), device=device
        )
        mb = self.cache.element_size() * self.cache.nelement() / 1024 ** 2
        print(f"Buffer pre-allocated: {mb:.2f} MB (one-time cost)")

    def update(self, new_token_embedding, index):
        """
        In-place slot write — no new memory created.
        If we used torch.cat instead, the GPU would re-allocate on every token.
        """
        self.cache[:, index : index + 1, :] = new_token_embedding


def simulate_inference(num_tokens=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, T, D = 1, 1024, 512

    kv_engine = StaticKVCache(B, T, D)
    print(f"\nGenerating {num_tokens} tokens with static KV cache...")

    t0 = time.perf_counter()
    for i in range(num_tokens):
        new_token = torch.randn(B, 1, D, device=device)
        kv_engine.update(new_token, i)
    elapsed = time.perf_counter() - t0

    print(f"Done: {num_tokens} tokens in {elapsed * 1000:.2f} ms")
    print("Engineering rule: Fragmentation is the enemy. Static buffers are the cure.")


if __name__ == "__main__":
    simulate_inference()
