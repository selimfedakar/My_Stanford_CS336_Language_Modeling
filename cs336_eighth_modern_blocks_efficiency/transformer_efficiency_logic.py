import torch
import torch.nn as nn


# My Goal: Understand the structural shift to Pre-Norm and GQA for efficiency.

class ModernTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=32):
        super().__init__()
        # "Modern solution is changing the position of the Norm (Pre-Norm)"
        self.ln_1 = nn.RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.ln_2 = nn.RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        # 1. Attention Path with Pre-Norm
        # "Moving the Norm before the block for stability"
        residual = x
        x = self.ln_1(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + attn_out

        # 2. MLP Path with Pre-Norm
        residual = x
        x = self.ln_2(x)
        x = self.mlp(x)
        return residual + x


# --- THE EFFICIENCY CHECK ---
def check_attention_complexity(q_heads=32, kv_heads_gqa=8, kv_heads_mqa=1):
    """
    "Standard attention is slow because it moves a lot of data."
    """
    print(f"Standard MHA: {q_heads} Q-heads, {q_heads} K-heads, {q_heads} V-heads. (Expensive!)")
    print(f"GQA (Grouped): {q_heads} Q-heads sharing {kv_heads_gqa} K/V groups. (Gold Standard)")
    print(f"MQA (Multi-Query): {q_heads} Q-heads sharing exactly {kv_heads_mqa} K/V. (Fastest)")


check_attention_complexity()
print("-" * 30)
print("I've successfully documented the Modern Transformer Block consensus.")