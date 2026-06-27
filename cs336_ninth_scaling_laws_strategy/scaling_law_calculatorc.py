# "Scaling is no longer just a math formula; it's an economic strategy."
# Goal: calculate Chinchilla-optimal data size for a given compute budget.

# Chinchilla rule of thumb: train on ~20 tokens per parameter.
# Most Kaplan-era models (GPT-3, etc.) were massively under-trained on data.
TOKENS_PER_PARAM = 20


def chinchilla_optimal(params_billions):
    """
    Given a model size (in billions of parameters), return the optimal token count.
    Chinchilla: N and D (tokens) should scale proportionally — both doubled, not just one.
    """
    return params_billions * TOKENS_PER_PARAM  # billions of tokens


def estimate_from_compute(compute_flops):
    """
    Given a compute budget (FLOPs), estimate the Chinchilla-optimal model size and token count.
    Derived from C ≈ 6 × N × D and N* ≈ D* (equal scaling):
      N* ≈ (C / 6)^0.5   (approximate — assumes N ≈ D in parameter count)
    """
    n_optimal_params = (compute_flops / 6) ** 0.5
    d_optimal_tokens = n_optimal_params * TOKENS_PER_PARAM
    return n_optimal_params, d_optimal_tokens


def check_inference_cost(model_size_a, model_size_b):
    """
    "Inference efficiency is where the real money is saved."
    A model 25x larger is roughly 25x more expensive to serve at scale.
    """
    ratio = model_size_a / model_size_b
    print(f"  {model_size_a}B vs {model_size_b}B: {ratio:.1f}x size → ~{ratio:.1f}x inference cost")


if __name__ == "__main__":
    print("--- Chinchilla Efficiency Report ---")
    for size in [7, 13, 70, 175]:
        tokens = chinchilla_optimal(size)
        print(f"  {size:>4}B model → optimal training: {tokens}B tokens")
    print("Note: Most Kaplan-style models were under-trained on data.\n")

    # Given a compute budget, what model + data split is optimal?
    compute_budget = 6e23  # ~GPT-3 compute (FLOPs)
    n_opt, d_opt = estimate_from_compute(compute_budget)
    print(f"--- Compute-Optimal Estimate (C = {compute_budget:.1e} FLOPs) ---")
    print(f"  Optimal model size: {n_opt / 1e9:.1f}B params")
    print(f"  Optimal token count: {d_opt / 1e9:.1f}B tokens")
    print()

    print("--- Inference Cost Comparison ---")
    check_inference_cost(175, 7)   # GPT-3 vs Llama-7B
    check_inference_cost(70,  7)   # Llama-70B vs Llama-7B
    print("-" * 30)
    print("Scaling is no longer just a math formula — it's an economic strategy.")
