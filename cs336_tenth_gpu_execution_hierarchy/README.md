# GPU Execution Model - The Massive Construction Project

Today, I demystified how my code actually runs on the hardware. I documented the GPU execution hierarchy using a "Construction Project" (İnşaat Projesi) analogy to understand how thousands of threads work in lockstep.

##  My Notes
![GPU Execution & Hardware Hierarchy](notes/page1.png)

##  The Execution Hierarchy (From Big to Small)
I analyzed how the software (Grid) maps to the hardware:
- **The Grid (The Whole Job):** This is the entire software kernel I launch.
- **The Thread Block (The Crews):** The grid is divided into smaller groups assigned to a specific **Streaming Multiprocessor (SM)**.
- **The Warp (The Squad):** The hardware executes 32 threads together in a group called a "Warp".
- **The Thread (The Worker):** A single execution path.

## ⚠ The "Branch Divergence" Trap
I discovered a "fantastic catch" about GPU performance: **Lockstep Execution**.
- Warps move in perfect synchronization. 
- **The Problem:** If I write `if (x) { ... } else { ... }`, and half the threads go into the `if` while the other half go into the `else`, the GPU slows down.
- **The Reality:** One half must pause while the other half runs, doubling the time. I must be careful with branching!

##  The GPU as a "Giant Warehouse"
I realized that if you look at a GPU chip diagram, almost everything is **Memory** (L1 Cache, Shared Memory, Register File), not calculation units.
- **ALUs (Arithmetic Logic Units):** These are the small "thinking parts" (math workers).
- **The Bottleneck:** Math is cheap; memory is expensive. The GPU is mostly a giant, complex warehouse where the slowest part is moving data to the workers.

##  Why FlashAttention is Revolutionary
Because it breaks the "Memory Bottleneck" by keeping data in the fast, small memory (SRAM) as much as possible, avoiding the expensive trips to the HBM "Warehouse".