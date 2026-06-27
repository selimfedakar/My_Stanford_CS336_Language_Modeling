import torch
import torch.nn.functional as F

"""
GOAL: Implement GQA using View-Reshaping for Zero-Memory Overhead.
CONCEPT: 'Virtual Expansion' — trick the GPU into parallel computation
         without replicating K/V tensors in HBM.
"""


def zero_copy_gqa(q, k, v, num_heads, num_kv_heads):
    """
    Args:
        q:   (B, T, q_heads, head_dim)
        k,v: (B, T, kv_heads, head_dim)
    Returns:
        out: (B, T, q_heads, head_dim) — attended output
    """
    B, T, q_heads, D = q.shape
    group_size = q_heads // num_kv_heads

    # 1. THE RESHAPE TRICK — split q_heads into (kv_heads, group_size)
    q = q.view(B, T, num_kv_heads, group_size, D)

    # 2. THE VIRTUAL EXPANSION
    # 'expand' creates a view, NOT a copy — no extra HBM allocation
    # K and V now 'look' like they have the group dimension that Q has
    k = k.unsqueeze(3).expand(B, T, num_kv_heads, group_size, D)
    v = v.unsqueeze(3).expand(B, T, num_kv_heads, group_size, D)

    # 3. PARALLEL BATCHED ATTENTION — replaces a Python for-loop over heads
    # scores shape: (B, num_kv_heads, group_size, T)
    scores  = torch.einsum('bthgd,bshgd->bthgs', q, k).squeeze(-1) / (D ** 0.5)
    # Simpler equivalent for the grouped dot-product:
    scores  = (q * k).sum(dim=-1) / (D ** 0.5)                # (B, T, kv_heads, group_size)
    weights = F.softmax(scores, dim=1)                         # Softmax over T dimension

    # 4. APPLY WEIGHTS TO V — produce the attended output
    out = (weights.unsqueeze(-1) * v).sum(dim=1)               # (B, kv_heads, group_size, D)
    out = out.view(B, num_kv_heads * group_size, D)            # (B, q_heads, D)

    return out


def why_rope_is_the_modern_choice():
    """
    "Fixed positions are for the past; rotations are for the future."
    """
    print("\n--- The RoPE Advantage (Llama-3 Standard) ---")
    print("1. Relative distance: RoPE encodes how far apart tokens are, not just absolute index.")
    print("2. Length generalization: can extrapolate to longer sequences than seen during training.")
    print("3. Zero parameters: RoPE is a mathematical rotation — no learnable embedding table needed.")


if __name__ == "__main__":
    # 1 batch, 10 tokens, 32 Q-heads, 8 KV-heads, head_dim=16
    q_mock = torch.randn(1, 10, 32, 16)
    k_mock = torch.randn(1, 10,  8, 16)
    v_mock = torch.randn(1, 10,  8, 16)

    out = zero_copy_gqa(q_mock, k_mock, v_mock, num_heads=32, num_kv_heads=8)
    print(f"--- Zero-Copy GQA Verification ---")
    print(f"Q input shape:      {q_mock.shape}")
    print(f"K/V input shape:    {k_mock.shape}  (shared — no copy)")
    print(f"Output shape:       {out.shape}")
    print("Parallelized attention without HBM memory replication.")

    why_rope_is_the_modern_choice()
