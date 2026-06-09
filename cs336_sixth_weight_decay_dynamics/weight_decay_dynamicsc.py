import torch

# "Weight decay has a secret second job: managing optimization dynamics."
# It's not just regularization — it actively shrinks weights at every step,
# counterbalancing the tendency of adaptive optimizers to let norms grow unbounded.


def sgd_weight_decay_step(weights, grads, lr, wd):
    """
    Coupled weight decay (L2 regularization style):
    w = w - lr * (grad + wd * w)
    The decay is folded into the gradient — the learning rate scales both.
    """
    with torch.no_grad():
        # "Shrink the weights slightly before applying the gradient nudge"
        weights -= lr * (grads + wd * weights)
    return weights


def adamw_weight_decay_step(weights, grads, lr, wd):
    """
    Decoupled weight decay (AdamW style):
    w = w - lr * grad - lr * wd * w
    Decay is applied independently from the gradient step.
    This is the 'secret second job': it keeps weight norms in check regardless
    of gradient magnitude — critical for adaptive optimizers like Adam.
    """
    with torch.no_grad():
        weights -= lr * grads        # Gradient step
        weights -= lr * wd * weights # Decay step — decoupled
    return weights


if __name__ == "__main__":
    torch.manual_seed(42)
    lr, wd = 0.1, 0.01
    steps  = 20

    # Track weight norm over multiple steps to see the decay dynamic
    w_sgd   = torch.randn(10)
    w_adamw = torch.randn(10).clone()
    g       = torch.randn(10) * 0.01  # Small gradients — decay dominates

    print(f"{'Step':<6} {'SGD+WD norm':<18} {'AdamW norm':<18}")
    print("-" * 42)
    for t in range(steps):
        w_sgd   = sgd_weight_decay_step(w_sgd,   g, lr, wd)
        w_adamw = adamw_weight_decay_step(w_adamw, g, lr, wd)
        if t % 5 == 0 or t == steps - 1:
            print(f"{t:<6} {w_sgd.norm().item():<18.4f} {w_adamw.norm().item():<18.4f}")

    print("-" * 42)
    print("Weight decay successfully managed the optimization dynamic.")
    print("AdamW decouples decay from gradient scaling — preferred for Transformer training.")
