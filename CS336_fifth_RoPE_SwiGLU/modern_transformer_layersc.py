import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# "Self-attention is position-blind... RoPE helps it understand the order."
# Consensus Rule: No bias terms, SwiGLU FFN, RoPE positional encoding.


# ─── SwiGLU Feed-Forward ─────────────────────────────────────────────────────
# "Swish(x @ W₁) * (x @ W₂) is the dominant architecture now."
# SwiGLU is a GATED unit: one branch applies Swish, the other is a linear gate.
# d_ff = 8/3 * d_model — scaled down from 4x because SwiGLU uses two projection matrices.

class SwiGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        d_ff = int(d_model * 8 / 3)  # Canonical SwiGLU hidden dim
        # No bias terms — consensus for modern LLMs
        self.W1 = nn.Linear(d_model, d_ff, bias=False)  # Gate branch (Swish)
        self.W2 = nn.Linear(d_model, d_ff, bias=False)  # Value branch
        self.W3 = nn.Linear(d_ff,    d_model, bias=False)  # Output projection

    def forward(self, x):
        # Swish(x @ W₁) * (x @ W₂) — the gate controls how much of W₂ passes through
        swish_gate = F.silu(self.W1(x))   # SiLU = Swish = x * sigmoid(x)
        value      = self.W2(x)
        return self.W3(swish_gate * value)


# ─── RoPE (Rotary Positional Embedding) ──────────────────────────────────────
# "Self-attention is position-blind" — without positional info, token order is lost.
# RoPE encodes position by rotating query/key vectors — relative distances are
# preserved in the dot product, which is what attention actually computes.

def precompute_rope_freqs(head_dim, max_seq_len, base=10000):
    """Precompute the cos/sin rotation matrices for RoPE."""
    # Frequencies: θᵢ = 1 / base^(2i / head_dim)
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    freqs = torch.outer(positions, theta)         # [seq_len, head_dim/2]
    return torch.cos(freqs), torch.sin(freqs)     # Each shape: [seq_len, head_dim/2]


def apply_rope(x, cos, sin):
    """
    Rotate each pair of dimensions (x₀, x₁) by the position angle.
    Rotation: [x₀·cos - x₁·sin,  x₀·sin + x₁·cos]
    """
    # Split into pairs of dimensions
    x1, x2 = x[..., ::2], x[..., 1::2]
    x_rotated = torch.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], dim=-1)
    return x_rotated.flatten(-2)  # Merge the last two dims back


if __name__ == "__main__":
    d_model   = 512
    head_dim  = 64
    seq_len   = 16
    batch     = 2

    # SwiGLU test
    ffn = SwiGLU(d_model)
    x   = torch.randn(batch, seq_len, d_model)
    out = ffn(x)
    print(f"SwiGLU — input: {x.shape} → output: {out.shape}")
    print(f"  d_ff (8/3 × d_model): {int(d_model * 8 / 3)}")

    # RoPE test
    cos, sin = precompute_rope_freqs(head_dim, seq_len)
    q = torch.randn(batch, seq_len, head_dim)
    q_rotated = apply_rope(q, cos, sin)
    print(f"\nRoPE — query shape unchanged: {q.shape} → {q_rotated.shape}")
    print("Consensus Rule: No bias terms, SwiGLU FFN, RoPE — a perfect balance.")
