import torch

# "RMSNorm skips these two steps... it's faster and requires less memory movement."
# I am implementing the consensus variant for modern LLMs.

def rms_norm(x, eps=1e-6):
    # I subtract nothing, I just normalize by the root mean square.
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

# Pre-Norm vs Post-Norm decision
# Consensus: Put the LayerNorm in FRONT of the block.
print("Architecture decision: Using Pre-Norm for smoother training gradients.")