import torch
import math

"""
GOAL: Simulate numerically correct Online Softmax using Tiling.
FOCUS: Memory-efficient computation — the algorithmic idea behind FlashAttention.

Bug fixed from original: when running_max updates, exp_values for previous tiles
must also be rescaled. Without this, numerator and denominator use different bases
and softmax outputs are wrong. Fix: rescale exp_values[:start] in-place when max changes.
Note: real FlashAttention avoids storing exp_values entirely — it recomputes them
during the backward pass instead of keeping O(N) intermediates.
"""


def tiled_online_softmax(x, tile_size):
    """
    Compute softmax(x) using tiled processing — single logical pass over data.

    Args:
        x:         1D tensor of attention scores
        tile_size: number of elements that fit into SRAM per tile
    Returns:
        softmax probabilities
    """
    N = x.shape[0]
    running_max = -float('inf')
    running_sum = 0.0
    exp_values  = torch.zeros_like(x)

    num_tiles = math.ceil(N / tile_size)
    print(f"Processing {N} elements in {num_tiles} tiles...\n")

    for t in range(num_tiles):
        start = t * tile_size
        end   = min((t + 1) * tile_size, N)
        tile  = x[start:end]

        local_max = tile.max().item()
        new_max   = max(running_max, local_max)

        if running_max != -float('inf') and new_max != running_max:
            # Max increased — rescale running_sum AND all previously stored exp_values
            scale = math.exp(running_max - new_max)
            running_sum        *= scale
            exp_values[:start] *= scale  # Keep numerator consistent with new global max

        exp_tile            = torch.exp(tile - new_max)
        running_sum        += exp_tile.sum().item()
        exp_values[start:end] = exp_tile
        running_max         = new_max

        print(f"  Tile {t + 1}/{num_tiles} | running_max={running_max:.4f} | running_sum={running_sum:.4f}")

    softmax = exp_values / running_sum
    print("\nCompleted — single pass, numerically stable.")
    return softmax


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(4096)

    tiled_result = tiled_online_softmax(x, tile_size=512)
    reference    = torch.softmax(x, dim=0)

    max_error = (tiled_result - reference).abs().max().item()
    print(f"\nMax error vs torch.softmax: {max_error:.2e}")
    assert max_error < 1e-5, "Softmax mismatch — check rescaling logic"
    print("Verification passed.")
