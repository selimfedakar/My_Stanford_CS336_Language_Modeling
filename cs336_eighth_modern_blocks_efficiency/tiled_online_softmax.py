import torch
import math

"""
GOAL: Simulate numerically correct Online Softmax using Tiling.
FOCUS: Memory-efficient computation (FlashAttention-style idea)
"""

def tiled_online_softmax(x, tile_size):
    """
    Compute softmax(x) using tiled processing.

    Args:
        x: 1D tensor (a row of attention scores)
        tile_size: how many elements fit into SRAM

    Returns:
        softmax vector
    """

    N = x.shape[0]

    # Global running values
    running_max = -float('inf')
    running_sum = 0.0

    # Store intermediate exp values (simulating recomputation possibility)
    exp_values = torch.zeros_like(x)

    num_tiles = math.ceil(N / tile_size)

    print(f"Processing {N} elements in {num_tiles} tiles...\n")

    for t in range(num_tiles):
        start = t * tile_size
        end = min((t + 1) * tile_size, N)

        tile = x[start:end]

        # Step 1: local max
        local_max = torch.max(tile).item()

        # Step 2: update global max
        new_max = max(running_max, local_max)

        # Step 3: rescale previous sum
        if running_max != -float('inf'):
            running_sum *= math.exp(running_max - new_max)

        # Step 4: compute exp for current tile
        exp_tile = torch.exp(tile - new_max)

        # Step 5: update sum
        running_sum += torch.sum(exp_tile).item()

        # Step 6: store intermediate exp values (optional)
        exp_values[start:end] = exp_tile

        running_max = new_max

        print(f"Tile {t+1}/{num_tiles} processed | running_max={running_max:.4f}")

    # Final normalization
    softmax = exp_values / running_sum

    print("\nCompleted with single pass over data (FlashAttention idea).")

    return softmax


# --- Demo ---
if __name__ == "__main__":
    torch.manual_seed(0)

    x = torch.randn(4096)

    # Tiled version
    tiled_result = tiled_online_softmax(x, tile_size=512)

    # Ground truth
    reference = torch.softmax(x, dim=0)

    # Compare
    max_error = torch.max(torch.abs(tiled_result - reference)).item()
    print(f"\nMax error vs torch.softmax: {max_error:.10f}")