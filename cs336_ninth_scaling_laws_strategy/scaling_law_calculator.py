# My Goal: Calculate the Chinchilla-optimal data size for a given compute budget.
# "Scaling is no longer just a math formula; it's an economic strategy."

def estimate_chinchilla_optimal(compute_budget_pflops):
    """
    Simulating the Chinchilla rule: N and D should scale proportionally.
    For a given compute budget, what is the best split?
    """
    # Simplified power-law constant: ~20 tokens per parameter
    tokens_per_param = 20

    def calculate_data(params_billions):
        return params_billions * tokens_per_param

    print(f"--- Chinchilla Efficiency Report ---")
    print(f"For a 7B model: Optimal data is {calculate_data(7)} Billion tokens.")
    print(f"For a 70B model: Optimal data is {calculate_data(70)} Billion tokens.")
    print(f"Note: Most older models (Kaplan-style) were under-trained.")


# --- THE ECONOMIC REALITY ---
# "Inference efficiency is where the real money is saved."
def check_inference_cost(model_size_a, model_size_b):
    ratio = model_size_a / model_size_b
    print(f"Model A is {ratio}x larger than Model B.")
    print(f"Model B will be roughly {ratio}x cheaper to serve in production.")


estimate_chinchilla_optimal(100)
print("-" * 30)
check_inference_cost(175, 7)  # Comparing GPT-3 size vs Llama-7B size