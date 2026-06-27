import torch

"""
GOAL: Demonstrate why subtracting the max prevents overflow.
CONCEPT: Stable Softmax — core requirement for Transformers at any precision.
"""


def simulate_numerical_explosion():
    print("--- Numerical Stability Analysis ---\n")

    # Large values — realistic for attention logits after many layers
    x_fp32 = torch.tensor([100.0, 101.0, 102.0], dtype=torch.float32)
    x_fp16 = x_fp32.to(torch.float16)

    # 1. NAIVE EXP — FP16 overflows, FP32 survives but barely
    print("Naive exp (FP32):")
    print(torch.exp(x_fp32))

    print("\nNaive exp (FP16):")
    print(torch.exp(x_fp16))  # inf — FP16 max exponent is ~88

    # 2. SAFE EXP — subtract max before exponentiation
    max_x = x_fp32.max()  # FP32 max used for both to preserve precision in subtraction
    print("\nSafe exp FP32 (after subtracting max):")
    print(torch.exp(x_fp32 - max_x))

    print("\nSafe exp FP16 (no overflow):")
    print(torch.exp(x_fp16 - max_x))

    # 3. FINAL SOFTMAX — mathematically identical, numerically safe
    safe_exp = torch.exp(x_fp32 - max_x)
    softmax  = safe_exp / safe_exp.sum()
    print("\nFinal Softmax (stable):")
    print(softmax)


def explain_stable_softmax():
    print("\n--- Stable Softmax Formula ---")
    print("""
  Softmax(xᵢ) = exp(xᵢ - max(x)) / Σⱼ exp(xⱼ - max(x))

  Why it works:
  - Prevents overflow: exp never sees large positive values
  - Largest value becomes exp(0) = 1 — anchors the distribution
  - Mathematically identical to naive softmax (max cancels in numerator/denominator)
""")


def precision_check():
    print("\n--- Precision Landscape ---")
    print("FP32: High precision, large memory cost")
    print("BF16: Same exponent range as FP32 — very stable (preferred for training)")
    print("FP16: Smaller exponent range → prone to overflow at large logit values")
    print("FP8 : Extremely efficient, requires careful per-tensor scaling")
    print("\nEngineering Rule: Lower precision → higher need for numerical stability tricks.")


if __name__ == "__main__":
    simulate_numerical_explosion()
    explain_stable_softmax()
    precision_check()
