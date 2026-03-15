# Memory Coalescing & The KV Cache Bottleneck

Today, I went deeper into the hardware reality of GPUs. I documented how the GPU fetches data in "chunky" bags and why inference (generating tokens) is a completely different beast compared to training.

##  My Notes
![Memory Coalescing & KV Cache](notes/page13_14.jpg)

##  Memory Coalescing: The Grocery Store Analogy
I realized that DRAM is actually very slow. When a GPU brings data from the main memory (HBM), it doesn't just grab a single number.
- **The Bag Logic:** It grabs a big chunk (usually 128 bytes). It's like a grocery store; you can't just buy one single grape, you have to buy the whole bag.
- **Coalesced Access:** If thread 1 asks for the 1st number and thread 2 asks for the 2nd (row-reading), the hardware delivers them in the same "bag" in one single trip. This is **Coalesced Access**.
- **The Column Trap:** If we read column-wise, each number is far away in memory, forcing the GPU to make many expensive trips for "mostly empty bags."

##  Tiling (Fayans Döşeme)
I documented the act of breaking a massive $N \times N$ matrix into small blocks called **Tiles** (usually $128 \times 128$).
- **The Goldilocks Zone:** - If the tile is too big, it won't fit in the SRAM and the code will crash.
  - If it's too small, we make too many trips to the memory and it becomes slow again.

##  Why Inference is Slow: The KV Cache
I identified why generating tokens is "massively waiting."
- **Training:** We process many tokens at once (batching).
- **Inference:** We generate only **one token at a time**. For every single token, we have to load the entire model weights from HBM.
- **The Solution:** I documented the **KV Cache**, which stores past Keys and Values so we don't have to recompute the entire history for every new word.