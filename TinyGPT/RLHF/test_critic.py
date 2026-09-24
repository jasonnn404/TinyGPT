from pathlib import Path

import torch

from TinyGPT.model import TinyGPT
from TinyGPT.bpe_tokenizer import BPETokenizer
from TinyGPT.RLHF.critic import Critic


device = "mps" if torch.backends.mps.is_available() else "cpu"

checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "sft_model.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)
config = checkpoint["config"]

tokenizer = BPETokenizer(vocab_size=config["bpe_vocab_size"])
tokenizer.load_merges(checkpoint["tokenizer_merges"])

backbone = TinyGPT(
    vocab_size=config["vocab_size"],
    max_sequence_length=config["max_sequence_length"],
    embedding_dimension=config["embedding_dimension"],
    num_heads=config["num_heads"],
    num_layers=config["num_layers"],
    dropout=config["dropout"]
).to(device)

backbone.load_state_dict(checkpoint["model_state_dict"])

critic = Critic(backbone).to(device)

text = "User: What is 1 + 1?\nAssistant: 1 + 1 is 2."
ids = tokenizer.encode(text)
input_ids = torch.tensor([ids], dtype=torch.long, device=device)

values = critic(input_ids)

print("input shape:", input_ids.shape)
print("values shape:", values.shape)
print("values:", values)
