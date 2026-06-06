import torch

# "Forward cost is 2 * (tokens * params). Backward is x2."
# Total rule of thumb: training cost ≈ 6 × tokens × params (FLOPs).


def show_activation_variance(label, weights, x):
    """Propagate x through a linear layer and report output statistics."""
    out = x @ weights
    print(f"  {label:30s} | input std: {x.std():.3f} | output std: {out.std():.3f}")


input_dim = 512

# Bad init: raw Gaussian — variance compounds with depth, activations explode
weights_bad  = torch.randn(input_dim, input_dim)

# Good init: scale by 1/sqrt(fan_in) — keeps output variance ≈ input variance
scale        = 1.0 / (input_dim ** 0.5)
weights_good = torch.randn(input_dim, input_dim) * scale

x = torch.randn(32, input_dim)  # Batch of 32 inputs


if __name__ == "__main__":
    print("--- Initialization: effect on activation variance ---")
    show_activation_variance("Unscaled (bad init)",    weights_bad,  x)
    show_activation_variance("Scaled 1/√fan_in (good)", weights_good, x)
    # Good init keeps output std ≈ 1.0; bad init amplifies it by √input_dim

    print(f"\nGood init weight std: {weights_good.std():.4f}  (≈ {scale:.4f} = 1/√{input_dim})")

    # --- 6x Rule: what does training cost look like at scale? ---
    # Example: GPT-3 — 175B params, 300B tokens
    params  = 175e9
    tokens  = 300e9
    flops   = 6 * tokens * params
    print(f"\n--- 6x Rule applied to GPT-3 ---")
    print(f"  Params:  {params / 1e9:.0f}B")
    print(f"  Tokens:  {tokens / 1e9:.0f}B")
    print(f"  FLOPs:   {flops:.2e}  (~{flops / 1e24:.1f} × 10²⁴)")
