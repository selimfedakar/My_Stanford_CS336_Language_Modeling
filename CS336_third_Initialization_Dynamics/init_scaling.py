import torch

# "Forward cost is 2 * (tokens * params). Backward is x2."
# Total rule of thumb: Training cost is roughly 6x tokens * params.

input_dim = 512
# If I use random Gaussian, output values become very large.
weights_bad = torch.randn(input_dim, input_dim)

# The solution from my notes: Scale your initial weights.
# 1 / sqrt(input_dim)
scale = 1.0 / (input_dim**0.5)
weights_good = torch.randn(input_dim, input_dim) * scale

print(f"Scaled weight std: {weights_good.std():.4f}")
print("This ensures output layers stay stable during training properly.")