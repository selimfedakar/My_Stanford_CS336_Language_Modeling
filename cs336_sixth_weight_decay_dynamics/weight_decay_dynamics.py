import torch

# My Goal: Understand weight decay as an optimization tool.
# "Weight decay has a secret second job: managing optimization dynamics."

def simulate_weight_decay_step(weights, grads, lr, wd):
    """
    Standard L2 Regularization / Weight Decay implementation.
    Formula: w = w - lr * (grad + wd * w)
    """
    with torch.no_grad():
        # "It's like shrinking the weights before the learning rate cools down."
        weights -= lr * (grads + wd * weights)
    return weights

# 1. Setup: Weights and Gradients
weights = torch.randn(10, requires_grad=True)
grads = torch.randn(10)
initial_norm = weights.norm().item()

# 2. Parameters (Pre-training context)
learning_rate = 0.1
weight_decay = 0.01 # My 'special dynamics' tool

# 3. Optimization Step
updated_weights = simulate_weight_decay_step(weights, grads, learning_rate, weight_decay)

print(f"Initial Weights Norm: {initial_norm:.4f}")
print(f"Updated Weights Norm: {updated_weights.norm().item():.4f}")
print("-" * 30)
print("Weight decay successfully managed the optimization dynamic.")