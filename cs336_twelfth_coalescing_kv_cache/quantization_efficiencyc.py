import torch

"""
GOAL: Understand how bit-width affects the memory bottleneck.
CONCEPT: Loading 4 items in one 'bag' instead of 1.
"""


def simulate_quantization_gain(rows=4096, cols=4096):
    """Compare FP32 vs INT8 memory footprint for the same weight matrix."""
    weights_fp32 = torch.randn(rows, cols)

    # INT8 quantization: scale to [-127, 127] range, clamp to avoid overflow
    scale        = weights_fp32.abs().max() / 127.0
    weights_int8 = (weights_fp32 / scale).round().clamp(-128, 127).to(torch.int8)

    mb_fp32 = weights_fp32.nelement() * weights_fp32.element_size() / 1024 ** 2
    mb_int8 = weights_int8.nelement() * weights_int8.element_size() / 1024 ** 2
    gain    = weights_fp32.element_size() / weights_int8.element_size()

    print("--- Weight Compression Report ---")
    print(f"FP32 weight size:          {mb_fp32:.2f} MB")
    print(f"INT8 weight size:          {mb_int8:.2f} MB")
    print(f"Theoretical bandwidth gain: {gain:.1f}x more data per HBM trip")
    return scale, weights_int8


def dequant_simulation(x_int8, scale):
    """
    "We store in INT8 (small bag), but compute in FP16 (big table)."
    Dequantization: cast back to float for ALU computation.
    Cost: one extra type cast — much cheaper than keeping weights in FP32.
    """
    return x_int8.to(torch.float16) * scale.to(torch.float16)


def final_scaling_check(model_name, params_b, precision_bits):
    """VRAM estimate for a given model size and precision."""
    vram_gb = params_b * (precision_bits / 8)
    print(f"\n--- Deployment Audit: {model_name} ---")
    print(f"Precision:       {precision_bits}-bit")
    print(f"Min VRAM needed: {vram_gb:.1f} GB")
    verdict = "Consumer-grade ready (RTX 3090/4090)" if vram_gb <= 24 else "Requires data-center GPU (H100/A100)"
    print(f"Status:          {verdict}")


if __name__ == "__main__":
    scale, weights_int8 = simulate_quantization_gain()

    # Show dequantization in action
    recon = dequant_simulation(weights_int8, scale)
    print(f"\nDequantized dtype: {recon.dtype}  shape: {recon.shape}")

    # VRAM audit for popular models
    final_scaling_check("Llama-3-8B   FP16",  8,  16)
    final_scaling_check("Llama-3-70B  FP16",  70, 16)
    final_scaling_check("Llama-3-70B  Q4",    70,  4)  # Current industry favorite
