# Stanford CS336 — Language Modeling from Scratch

<p align="center">
  <img src="https://raw.githubusercontent.com/selimfedakar/My_Stanford_CS336_Language_Modeling/main/CS336_first_Efficiency_Tokenization/notes/page1.jpg" width="540" alt="Hand-written notes — Efficiency & Tokenization" />
</p>

<p align="center"><i>Real understanding, written by hand. One topic, one notebook page, one code block at a time.</i></p>

---

My personal deep-dive into Stanford's CS336: Language Modeling from Scratch.

This is not a homework dump. I start every module with hand-written notes — diagrams, derivations, my own intuitions — before I write a single line of code. The notes are the primary artifact. The code is my proof that I actually understood the concept.

---

## Why hand-written notes?

In a world full of AI-generated explanations and copy-pasted code, the hardest thing to fake is a page of hand-written math. It is proof that I sat with the problem long enough to internalize it.

I have been doing this for over a year. Every module in this repo follows the same ritual: I read the paper or lecture, I understand it deeply enough to explain it by hand, then I build the key idea in code myself.

---

## My Areas of Focus

I started from the question most people skip: why does my code run slow, and where exactly does time go on a GPU? That pulled me into hardware — MFU, memory bandwidth, the difference between compute-bound and memory-bound operations. From there I worked my way up through the full stack myself.

I dug into memory layout: how tensors actually live in RAM, what strides are, why a transpose can silently kill performance. Then I tackled initialization and training dynamics — the math behind why weights need to be set a certain way before a single gradient step, and what breaks when you get it wrong.

On the architecture side I analyzed the decisions that separate modern models from older ones: why RMSNorm replaced LayerNorm, how RoPE encodes position without adding parameters, why SwiGLU became the default activation. I treated each of these as a design question I had to answer for myself, not a given I had to memorize.

Scaling laws were a turning point. I documented the shift from Kaplan to Chinchilla — from "scale the model" to "scale the data too" — and worked through the realization that training large models is as much an economic strategy as a technical one.

The second half of this repo is where I went deep into GPU programming: the CUDA execution model (grid, block, warp), kernel fusion, FlashAttention's core insight that doing more math to move less data is actually faster, memory coalescing, KV cache mechanics, and finally PTX-level parallelism. I built my understanding layer by layer until I could trace what happens between writing `tensor.matmul()` and electrons moving on silicon.

---

## How each module is structured

```
cs336_nth_topic/
├── README.md          my hand-written notes reconstructed in writing
├── notes/             scanned notebook pages — the real work
│   ├── page1.jpg
│   └── page2.jpg
└── topic.py           my implementation
```

Every README in this repo is a written reconstruction of my notes — not documentation, not a tutorial. It is a personal account of what I learned, what surprised me, and how I ended up thinking about it.

---

## What this covers

CS336 goes much deeper than "how to use a transformer." I built everything from first principles:

- **Hardware awareness** — why code runs slow, what MFU means, how to read a roofline chart
- **Memory systems** — strides, contiguous tensors, coalescing, preallocation
- **Architecture decisions** — why RMSNorm replaced LayerNorm, why SwiGLU became standard
- **Training science** — initialization theory, warmup schedules, weight decay without overfitting
- **Scaling economics** — Kaplan vs. Chinchilla, the real cost of compute-optimal training
- **GPU programming** — CUDA execution model, kernel fusion, FlashAttention from intuition to implementation
- **Low-level optimization** — PTX, parallel reductions, synchronization primitives

---

## Stack

| Area | Tools |
|------|-------|
| Language | Python |
| Deep learning | PyTorch |
| GPU programming | CUDA / Triton |
| Notation | Hand-written (pen + paper) |
| Reference | Stanford CS336 lecture materials |

---

## Status

| Component | Status |
|-----------|--------|
| Modules 01–16 | complete |
| Hand-written notes | complete (all modules) |
| Supporting code | complete |
| Repo in progress | yes — more modules coming |

---

## Author

**Ahmet Selim Fedakar** — Los Angeles  
Building things at the intersection of AI systems and real engineering.

---

> *"The best way to understand something is to be able to explain it by hand."*
