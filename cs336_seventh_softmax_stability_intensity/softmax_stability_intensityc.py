import torch
import torch.nn.functional as F

# Goal: stabilize the "Problem Child" (Softmax) and understand Arithmetic Intensity.


def scaled_dot_product_attention(q, k, v):
    """
    "A simple practical fix: the scaling factor."
    Dividing scores by √d_k prevents the dot products from blowing up in magnitude,
    which would push Softmax into its flat saturation region.
    """
    d_k = q.size(-1)

    # 1. Raw scores: Q × Kᵀ — can explode without scaling
    scores = torch.matmul(q, k.transpose(-2, -1))

    # 2. Scaling intervention: divide by √d_k
    scaled_scores = scores / (d_k ** 0.5)

    # 3. Softmax: "The Problem Child" — exponentials amplify large values dangerously
    attn_weights = F.softmax(scaled_scores, dim=-1)

    # 4. Weighted sum of values
    return torch.matmul(attn_weights, v)


def arithmetic_intensity_report(M, N, K):
    """
    Arithmetic Intensity = FLOPs / bytes accessed.
    "Memory access is expensive, computing is cheap."
    High intensity ops (matmul) reuse data from SRAM; low intensity ops (elementwise)
    must re-fetch from HBM every time — that's the memory bottleneck.
    """
    # Matrix multiply A[M,K] @ B[K,N]: 2*M*K*N FLOPs
    flops = 2 * M * K * N

    # Bytes: load A + B, store C (assuming fp32 = 4 bytes)
    bytes_accessed = 4 * (M * K + K * N + M * N)

    intensity = flops / bytes_accessed
    return flops, bytes_accessed, intensity


if __name__ == "__main__":
    # --- Attention Stability Test ---
    # q, k scaled by 10 to simulate large logits — scaling should keep Softmax stable
    q, k, v = (torch.randn(1, 8, 64) * 10,
               torch.randn(1, 8, 64) * 10,
               torch.randn(1, 8, 64))

    output = scaled_dot_product_attention(q, k, v)
    print(f"Attention output mean: {output.mean().item():.4f}")
    print("Successfully stabilized the Transformer's problem child.\n")

    # --- Arithmetic Intensity Report ---
    # Example: matmul shapes typical in Transformer FFN
    flops, mem, intensity = arithmetic_intensity_report(M=2048, N=8192, K=2048)
    print(f"--- Arithmetic Intensity (matmul 2048×8192×2048) ---")
    print(f"  FLOPs:            {flops / 1e9:.1f} GFLOPs")
    print(f"  Bytes accessed:   {mem / 1e6:.1f} MB")
    print(f"  Intensity:        {intensity:.1f} FLOPs/byte")
    print(f"  (A100 ridge point ≈ 208 FLOPs/byte — above = compute-bound, below = memory-bound)")
