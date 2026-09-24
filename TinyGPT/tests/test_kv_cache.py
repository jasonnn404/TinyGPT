from pathlib import Path

import torch

from TinyGPT.model import TinyGPT
from TinyGPT.bpe_tokenizer import BPETokenizer
from TinyGPT.generation import (
    generate_greedy,
    generate_greedy_cached,
    generate_greedy_preallocated
)


# -------------------------
# Device
# -------------------------

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)


# -------------------------
# Load checkpoint
# -------------------------

checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "sft_model.pt"

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

config = checkpoint["config"]


# -------------------------
# Load tokenizer
# -------------------------

tokenizer = BPETokenizer(
    vocab_size=config["vocab_size"]
)

tokenizer.load_merges(
    checkpoint["tokenizer_merges"]
)


# -------------------------
# Load model
# -------------------------

model = TinyGPT(
    vocab_size=config["vocab_size"],
    max_sequence_length=config["max_sequence_length"],
    embedding_dimension=config["embedding_dimension"],
    num_heads=config["num_heads"],
    num_layers=config["num_layers"],
    dropout=config["dropout"]
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# -------------------------
# Input
# -------------------------

prompt = "hello there"

input_ids = torch.tensor(
    [tokenizer.encode(prompt)],
    dtype=torch.long,
    device=device
)

print("\nPrompt:")
print(prompt)

print("\nInput IDs:")
print(input_ids)

print("\nInput shape:")
print(input_ids.shape)


# -------------------------
# Normal generation
# -------------------------

normal = generate_greedy(
    model,
    input_ids,
    max_new_tokens=20
)


# -------------------------
# KV-cache generation
# -------------------------

cached = generate_greedy_cached(
    model,
    input_ids,
    max_new_tokens=20
)


# -------------------------
# Preallocated KV-cache generation
# -------------------------

preallocated = generate_greedy_preallocated(
    model,
    input_ids,
    max_new_tokens=20
)

# -------------------------
# Decode results
# -------------------------

normal_ids = normal[0].tolist()
cached_ids = cached[0].tolist()

print("\n-------------------------")
print("Normal token IDs")
print("-------------------------")
print(normal_ids)

print("\n-------------------------")
print("KV-cache token IDs")
print("-------------------------")
print(cached_ids)


# -------------------------
# Correctness check
# -------------------------

print("\n-------------------------")
print("Correctness")
print("-------------------------")

naive_same = torch.equal(normal, cached)
preallocated_same = torch.equal(normal, preallocated)

print("Naive KV cache matches baseline:", naive_same)
print("Preallocated KV cache matches baseline:", preallocated_same)

if naive_same and preallocated_same:
    print("PASS: All generation methods produce identical tokens.")
else:
    print("FAIL: Generation outputs differ.")

    print("\nBaseline:")
    print(normal)

    print("\nNaive KV:")
    print(cached)

    print("\nPreallocated KV:")
    print(preallocated)
