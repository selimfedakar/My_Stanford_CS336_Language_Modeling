# Modern Transformer Blocks - Pre-Norm & Attention Efficiency

##  My Notes
![Pre-Norm & Attention Efficiency](notes/page1.png)

Today, I moved from the 2017 "Attention is All You Need" vanilla architecture to the modern consensus used in Llama and GPT models. I documented why simply stacking layers isn't enough and how we solve the "Quadratic Bottleneck" of attention.

##  The Shift to Pre-Norm
I identified a critical stability issue in older architectures where training would suddenly "blow up."
- **The Old Rule:** Post-Norm (Normalizing after the residual addition).
- **The Modern Solution:** Pre-Norm. I am now moving the Normalization (LayerNorm/RMSNorm) to the *front* of the blocks.
- **The Result:** This ensures much smoother gradient flow and allows for training significantly deeper models without instability.

## Solving the Quadratic Bottleneck (MQA & GQA)
I documented that standard attention is inherently **quadratic** ($O(N^2)$), meaning the cost explodes as the sequence length increases. I explored two "cheap" efficiency solutions:
1. **MQA (Multi-Query Attention):** All 32 Query heads share a single Key and Value head. It is extremely fast for inference but can lose some model quality.
2. **GQA (Grouped-Query Attention):** The modern "Gold Standard" balance used in Llama. Queries are grouped (e.g., 32 Queries share 8 K/V groups) to maintain performance while drastically reducing memory traffic.