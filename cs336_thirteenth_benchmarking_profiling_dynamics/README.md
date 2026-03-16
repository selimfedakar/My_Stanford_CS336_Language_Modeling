# Benchmarking vs. Profiling & The Data-Starved Reality

Today, I explored the "Gold Standard" of performance measurement. I documented how to identify bottlenecks in high-performance code and why modern AI is shifting toward smaller, more data-rich models.

##  My Notes
![Benchmarking vs Profiling](notes/page1.png)

##  Benchmarking vs. Profiling: The Drill Down
I realized that measuring speed isn't just about starting a timer.
- **Benchmarking (Wall-clock):** Measuring the total end-to-end performance of a task. It tells me *if* the code is slow.
- **Profiling (Drill Down):** Identifying exactly *where* the time is being spent. It tells me *why* the code is slow. 
- **The Question:** "Is the GPU waiting for data from HBM (memory-bound) or is it actually doing math (compute-bound)?"

##  The "Data-Starved" Shift ($D >> P$)
I documented a major trade-off in LLM training:
- **The Discovery:** Most models were actually **"under-trained"** (yetersiz eğitilmiş). They were compute-hungry but data-starved.
- **The Strategy:** We now feed smaller, faster models with **massive** amounts of data. This is much more efficient because inference becomes faster and cheaper while intelligence remains high.

##  Discretization (Ayrıklaştırma)
I revisited the technical term for transforming continuous attributes into discrete features.
- **Example:** Tokenization is essentially discretizing language into a finite vocabulary of integers.
- **Example 2:** Bucketing continuous data like age (0-18, 19-35, etc.) to help the model find patterns.