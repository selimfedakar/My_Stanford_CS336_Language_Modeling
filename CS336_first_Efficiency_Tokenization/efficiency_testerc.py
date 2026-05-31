import time

# "Accuracy = Efficiency x Resources"
# The combination of hardware and algorithms is what defines modern AI.
# MFU (Model FLOPs Utilization) = actual throughput / peak theoretical throughput


def calculate_mfu(achieved_flops, peak_flops):
    """
    Model FLOPs Utilization: how much of the hardware's theoretical peak we actually use.
    A100 peak ~312 TFLOP/s — real training typically achieves 35-55% MFU.
    Anything above 50% is considered good.
    """
    return achieved_flops / peak_flops


def tokenize(text, vocab):
    """
    Greedy subword tokenizer — demonstrates why BPE beats word-level tokenization.
    Tries longest match first; falls back to character-level for unknowns.
    """
    tokens = []
    i = 0
    while i < len(text):
        matched = False
        for length in range(len(text) - i, 0, -1):  # Longest match first
            candidate = text[i:i + length]
            if candidate in vocab:
                tokens.append(candidate)
                i += length
                matched = True
                break
        if not matched:
            tokens.append(text[i])  # Character-level fallback
            i += 1
    return tokens


if __name__ == "__main__":
    # --- Tokenization Analysis ---
    # Word-based: vocab size grows unbounded with every new word form
    word_vocab  = {"learning": 1, "learns": 2, "learner": 3}

    # BPE (subword): reuse "learn" across all derived forms — vocab stays bounded
    subword_vocab = {"learn": 1, "ing": 2, "s": 3, "er": 4}

    test_words = ["learning", "learns", "learner"]
    for word in test_words:
        subword_tokens = tokenize(word, subword_vocab)
        in_word_vocab  = word in word_vocab
        print(f"  '{word}' → subword: {subword_tokens} | in word vocab: {in_word_vocab}")

    print(f"\nBPE efficiency: 3 word forms covered by {len(subword_vocab)} sub-tokens.")
    print("Word-level vocab grows with every inflection; BPE reuses roots.\n")

    # --- MFU Calculation ---
    # Example: A100 GPU, achieved ~180 TFLOP/s on a training run
    peak_flops     = 312e12   # A100 BF16 peak: 312 TFLOP/s
    achieved_flops = 180e12   # Realistic training throughput
    mfu = calculate_mfu(achieved_flops, peak_flops)
    print(f"Achieved TFLOP/s: {achieved_flops / 1e12:.0f}")
    print(f"Peak TFLOP/s:     {peak_flops / 1e12:.0f}")
    print(f"MFU:              {mfu:.1%}  ({'good' if mfu > 0.5 else 'room to improve'})")
