# Scaling Laws - Kaplan vs. Chinchilla (The Economic Strategy)

Today, I moved from model architecture to the economics of AI. I documented that scaling a Large Language Model is no longer just a "math formula"—it has become a high-stakes **economic strategy**.

##  My Notes
![Scaling Laws & Strategy](notes/page1.png)

##  The Shift in Philosophy: Kaplan vs. Chinchilla
I analyzed the two dominant papers that shaped the current LLM landscape:

- **The Kaplan Rule (OpenAI, 2020):** This was the first paper to show that training LLMs is not random. It suggested that **model size ($N$)** is the most important factor. This led to "chunky" (hantal) models like GPT-3 that were massive but under-trained.
- **The Chinchilla Rule (DeepMind, 2022):** Proved that we are actually "data-constrained." They realized that if you have 10% more compute budget, you should split it equally: **5% more model size and 5% more data**.

##  Scaling as an Economic Strategy
I documented a key insight: We use massive amounts of data to squeeze every "drop of intelligence" out of smaller models.
- **Why?** Because **Inference Efficiency** is where the real money is saved. 
- **The Goal:** Smaller, well-trained models are significantly cheaper and faster to run in production for millions of users.

##  The Takeaway
Training is a one-time cost; inference is forever. My goal is to build "compute-optimal" models that respect the Chinchilla limits.