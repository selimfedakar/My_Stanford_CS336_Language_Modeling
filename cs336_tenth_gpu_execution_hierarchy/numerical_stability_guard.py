import torch

"""
GOAL: Demonstrate why subtracting the max prevents overflow.
CONCEPT: Stable Softmax = core requirement for Transformers.
"""

def simulate_numerical_explosion():
    print("--- Numerical Stability Analysis ---\n")

    # Large values (realistic for attention logits)
    x_fp32 = torch.tensor([100.0, 101.0, 102.0], dtype=torch.float32)
    x_fp16 = x_fp32.to(torch.float16)

    # --- 1. NAIVE EXP (FP32 vs FP16) ---
    naive_exp_fp32 = torch.exp(x_fp32)
    naive_exp_fp16 = torch.exp(x_fp16)

    print("Naive exp (FP32):")
    print(naive_exp_fp32)

    print("\nNaive exp (FP16):")
    print(naive_exp_fp16)  # Likely inf

    # --- 2. SAFE EXP (STABILIZED) ---
    max_x = torch.max(x_fp32)

    safe_exp_fp32 = torch.exp(x_fp32 - max_x)
    safe_exp_fp16 = torch.exp(x_fp16 - max_x)

    print("\nSafe exp (after subtracting max):")
    print(safe_exp_fp32)

    print("\nSafe exp FP16 (no overflow):")
    print(safe_exp_fp16)

    # --- 3. FINAL SOFTMAX ---
    softmax = safe_exp_fp32 / torch.sum(safe_exp_fp32)

    print("\nFinal Softmax (Stable):")
    print(softmax)


# --- THE CORE FORMULA ---
def explain_stable_softmax():
    print("\n--- Stable Softmax Formula ---")
    print("""
Softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))

Why it works:
- Prevents overflow (exp never sees huge values)
- Keeps largest value at exp(0) = 1
- Preserves exact probabilities (mathematically identical)
""")


# --- THE MACRO VIEW: PRECISION ---
def precision_check():
    print("\n--- Precision Landscape ---")

    print("FP32 : High precision, large memory cost")
    print("BF16 : Same exponent range as FP32 (very stable)")
    print("FP16 : Smaller range → prone to overflow")
    print("FP8  : Extremely efficient, requires careful scaling")

    print("\nEngineering Rule:")
    print("Lower precision → higher need for numerical stability tricks")


if __name__ == "__main__":
    simulate_numerical_explosion()
    explain_stable_softmax()
    precision_check()