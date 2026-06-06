import torch
import torch.nn as nn

# "RMSNorm skips these two steps... it's faster and requires less memory movement."
# The two steps LayerNorm does that RMSNorm drops:
#   1. Subtract the mean (center the distribution)
#   2. Learn an additive bias β
# RMSNorm only divides by the root mean square — simpler, fewer memory ops.


def layer_norm(x, gamma, beta, eps=1e-5):
    """Standard LayerNorm: center + scale + shift."""
    mean = x.mean(-1, keepdim=True)
    var  = x.var(-1, keepdim=True, unbiased=False)
    x_hat = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_hat + beta


def rms_norm(x, gamma, eps=1e-6):
    """
    RMSNorm: skip mean subtraction and additive bias — just normalize by RMS.
    torch.rsqrt = 1/sqrt, which fuses the reciprocal and sqrt into one op.
    """
    rms   = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return gamma * x * rms


if __name__ == "__main__":
    # Architecture decision: Pre-Norm (norm before the block) is the modern consensus.
    # Pre-Norm gives smoother gradient flow at depth; Post-Norm can destabilize early training.
    print("Architecture decision: Using Pre-Norm for smoother training gradients.\n")

    batch, seq, dim = 2, 8, 512
    x     = torch.randn(batch, seq, dim)
    gamma = torch.ones(dim)
    beta  = torch.zeros(dim)

    out_ln  = layer_norm(x, gamma, beta)
    out_rms = rms_norm(x, gamma)

    print(f"Input         — mean: {x.mean():.4f}  std: {x.std():.4f}")
    print(f"LayerNorm out — mean: {out_ln.mean():.4f}  std: {out_ln.std():.4f}")
    print(f"RMSNorm out   — mean: {out_rms.mean():.4f}  std: {out_rms.std():.4f}")
    print(f"\nRMSNorm skips mean subtraction → output mean ≠ 0, but variance is controlled.")
    print(f"For large models, skipping mean + bias saves memory bandwidth at every layer.")
