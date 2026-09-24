import time

import torch

from model import TinyGPT
from bpe_tokenizer import BPETokenizer
from generation import (
    generate_greedy,
    generate_greedy_cached,
    generate_greedy_preallocated
)


# --------------------------------------------------
# Device
# --------------------------------------------------

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)


def synchronize():
    """Make sure GPU work has actually finished before reading the timer."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


# --------------------------------------------------
# Load checkpoint
# --------------------------------------------------

checkpoint_path = "checkpoints/sft_model.pt"

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

config = checkpoint["config"]


# --------------------------------------------------
# Load tokenizer
# --------------------------------------------------

tokenizer = BPETokenizer(
    vocab_size=config["bpe_vocab_size"]
)

tokenizer.load_merges(
    checkpoint["tokenizer_merges"]
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = TinyGPT(
    vocab_size=config["vocab_size"],
    max_sequence_length=config["max_sequence_length"],
    embedding_dimension=config["embedding_dimension"],
    num_heads=config["num_heads"],
    num_layers=config["num_layers"],
    dropout=config["dropout"]
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# --------------------------------------------------
# Prompt
# --------------------------------------------------

prompt = "hello there"

input_ids = torch.tensor(
    [tokenizer.encode(prompt)],
    dtype=torch.long,
    device=device
)

prompt_length = input_ids.shape[1]
max_sequence_length = config["max_sequence_length"]

print("Prompt:", prompt)
print("Prompt tokens:", prompt_length)
print("Max sequence length:", max_sequence_length)


# --------------------------------------------------
# Benchmark helper
# --------------------------------------------------

def benchmark(fn, max_new_tokens, runs=5):
    # Warmup
    for _ in range(2):
        fn(
            model,
            input_ids,
            max_new_tokens=max_new_tokens
        )

    synchronize()

    times = []

    for _ in range(runs):
        synchronize()
        start = time.perf_counter()

        fn(
            model,
            input_ids,
            max_new_tokens=max_new_tokens
        )

        synchronize()
        end = time.perf_counter()

        times.append(end - start)

    return sum(times) / len(times)


# --------------------------------------------------
# Benchmark
# --------------------------------------------------

test_lengths = [1, 5, 10, 20, 50, 100, 200]

# We cannot exceed the model's learned position embedding range.
test_lengths = [
    n for n in test_lengths
    if prompt_length + n - 1 <= max_sequence_length
]

print("\nBenchmarking...")
print()

print(
    f"{'Tokens':<10}"
    f"{'Baseline':<15}"
    f"{'Naive KV':<15}"
    f"{'Prealloc KV':<15}"
    f"{'Naive':<10}"
    f"{'Prealloc':<10}"
)

print("-" * 75)

for num_tokens in test_lengths:
    baseline_time = benchmark(generate_greedy, num_tokens)
    naive_time = benchmark(generate_greedy_cached, num_tokens)
    prealloc_time = benchmark(generate_greedy_preallocated, num_tokens)

    naive_speedup = baseline_time / naive_time
    prealloc_speedup = baseline_time / prealloc_time

    print(
        f"{num_tokens:<10}"
        f"{baseline_time * 1000:<15.2f}"
        f"{naive_time * 1000:<15.2f}"
        f"{prealloc_time * 1000:<15.2f}"
        f"{naive_speedup:<10.2f}"
        f"{prealloc_speedup:<10.2f}"
    )