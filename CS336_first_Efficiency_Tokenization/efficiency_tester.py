import time

# "Accuracy = Efficiency x Resources"
# I realized that the combination of hardware and algorithms is what defines modern AI.

def calculate_training_potential(flops, efficiency):
    # My goal: Understand the promised speed vs actual wall-clock time.
    actual_performance = flops * efficiency
    return actual_performance

# Tokenization Analysis from my notes
# Word-based: Vocab size gets unbounded.
# BPE: Subword tokenization is the consensus.
vocab = {"learning": 1, "learns": 2, "learner": 3} # Unbounded problem
subword_vocab = {"learn": 1, "ing": 2, "s": 3, "er": 4} # BPE solution

print(f"BPE efficiency: I can represent 3 words with only 4 sub-tokens.")
print(f"Potential: {calculate_training_potential(100, 0.5)} MFU (Model FLOPs Utilization)")