import torch
import torch.nn as nn

# Goal: understand the structural shift to Pre-Norm and GQA for efficiency.
# "Modern solution is changing the position of the Norm (Pre-Norm)"


class ModernTransformerBlock(nn.Module):
    """
    Pre-Norm Transformer block — the current consensus architecture.
    Norm goes BEFORE each sub-block, not after: this stabilizes gradients at depth.
    Note: nn.RMSNorm requires PyTorch >= 2.4; swap with a custom RMSNorm for older versions.
    Note: MLP uses GELU here for clarity; SwiGLU (8/3 × d_model hidden dim) is preferred
          in production (see cs336_fifth/modern_transformer_layersc.py).
    """

    def __init__(self, dim, num_heads=8, batch_first=True):
        super().__init__()
        self.ln_1 = nn.RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=batch_first)
        self.ln_2 = nn.RMSNorm(dim)
        self.mlp  = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x):
        # 1. Attention path — Pre-Norm: norm first, then attend, then residual
        residual = x
        attn_out, _ = self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x))
        x = residual + attn_out

        # 2. MLP path — same Pre-Norm pattern
        residual = x
        x = residual + self.mlp(self.ln_2(x))
        return x


def check_attention_complexity(q_heads=32, kv_heads_gqa=8, kv_heads_mqa=1):
    """
    "Standard attention is slow because it moves a lot of data."
    GQA reduces KV cache memory by sharing K/V heads across query groups.
    """
    print(f"Standard MHA:  {q_heads} Q / {q_heads} K / {q_heads} V  (expensive — full KV cache)")
    print(f"GQA (Grouped): {q_heads} Q sharing {kv_heads_gqa} K/V groups  (Gold Standard — Llama-3, Mistral)")
    print(f"MQA:           {q_heads} Q sharing {kv_heads_mqa} K/V  (fastest — extreme memory savings)")


if __name__ == "__main__":
    check_attention_complexity()
    print("-" * 50)

    # Forward pass test
    batch, seq, dim = 2, 16, 256
    block = ModernTransformerBlock(dim, num_heads=8)
    x     = torch.randn(batch, seq, dim)
    out   = block(x)
    print(f"\nModernTransformerBlock — input: {x.shape} → output: {out.shape}")
    print("Successfully documented the Modern Transformer Block consensus.")
