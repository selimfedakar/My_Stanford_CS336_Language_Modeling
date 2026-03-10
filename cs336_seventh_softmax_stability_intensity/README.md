# Softmax - The Problem Child & Arithmetic Intensity

I have identified the "problem child" (sorunlu çocuk) of stability in Transformers: the **Softmax function**. Today, I documented why we need a "practical fix" (müdahale) to keep our models from exploding and why memory access is the real bottleneck of GPU performance.

##  My Notes
![Softmax Stability & Arithmetic Intensity](notes/page1.png)

## The "Problem Child": Softmax Stability
Because Softmax involves exponentials ($e^x$), the numbers inside the function ($Q \cdot K^T$) can get very large, which can make the model **"blow up" (patlamak)** and become unstable.
- **The Intervention:** I implemented the scaling factor from the original 'Attention is All You Need' paper.
- **The Scaling Factor:** By simply dividing the math by $\sqrt{d_k}$, we keep the scores in a safe range.
- **Double Trouble:** I documented that this "problem child" exists in two places: at the very end (output softmax) and inside every self-attention layer.

##  Arithmetic Intensity (Math vs. Memory)
I realized that to understand GPU performance, I must track **Arithmetic Intensity**.
- **The Reality:** Memory access is very expensive (slow) on a GPU, while computing is relatively cheap (fast).
- **The Goal:** We want high arithmetic intensity—doing a lot of compute for every single memory access we perform.
- **Hardware Awareness:** In modern AI hardware, "math is cheap, memory is expensive".

## Architectural Evolution: Pre-Norm
I am now moving toward the **Pre-Norm** consensus (unlike the 2017 original). By placing the LayerNorm *before* the softmax and attention blocks, we further stabilize the training flow of massive models.