import torch
import torch.nn.functional as F


# My Goal: Stabilize the "Problem Child" (Softmax) and understand Arithmetic Intensity.

def scaled_dot_product_attention(q, k, v):
    """
    "A simple practical fix: the scaling factor."
    """
    d_k = q.size(-1)

    # 1. Calculate scores: Q * K^T
    # "Without scaling, these numbers can blow up (patlamak)."
    scores = torch.matmul(q, k.transpose(-2, -1))

    # 2. Scaling Intervention: Divide by sqrt(d_k)
    scaled_scores = scores / (d_k ** 0.5)

    # 3. Softmax: "The Problem Child"
    # "Exponentials make this dangerous if scores are too large."
    attn_weights = F.softmax(scaled_scores, dim=-1)

    # 4. Contextual output
    return torch.matmul(attn_weights, v)


# --- HARDWARE REALITY CHECK ---
# "Memory access is expensive, computing is cheap."
def arithmetic_intensity_demo():
    # Doing a lot of math (cheap) on a small piece of data
    x = torch.randn(1024, 1024)
    # High intensity: many ops on same data in SRAM
    result = torch.mm(x, x).pow(2).sum()
    print(f"High Intensity Op Done: {result.item():.2f}")


# Simulation
q, k, v = torch.randn(1, 8, 64) * 10, torch.randn(1, 8, 64) * 10, torch.randn(1, 8, 64)
output = scaled_dot_product_attention(q, k, v)

print(f"Attention Output Mean: {output.mean().item():.4f}")
print("I've successfully stabilized the Transformer's problem child.")