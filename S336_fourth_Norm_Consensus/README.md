# 🛡️ Normalization Consensus - RMSNorm vs LayerNorm

## My Notes
![Normalization Comparison](notes/page1.png)
![](notes/page2.png)

In modern Large Language Models, small differences in architecture can lead to massive efficiency gains. Today, I explored why the industry switched to **RMSNorm**.

## ⚡ Why RMSNorm?
I documented two smart reasons for this change:
1. **Simplicity:** RMSNorm only normalizes the data; it doesn't subtract the mean or add a bias term like LayerNorm.
2. **Speed:** Because it is simpler, it is faster and requires less memory movement—which is just as important as FLOPs for speed.

## 🏗️ Pre-Norm Architecture
Following the Stanford lessons, I am implementing the **Pre-Norm** strategy:
- I put the layer norm **in front** of the block.
- I found that this "pre-norm" approach makes the training significantly smoother compared to older "post-norm" methods.
