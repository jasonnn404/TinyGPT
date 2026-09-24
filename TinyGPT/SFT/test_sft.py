from pathlib import Path

import torch

from TinyGPT.model import TinyGPT
from TinyGPT.bpe_tokenizer import BPETokenizer


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)


# ------------------------------------------------------------
# Load checkpoint
# ------------------------------------------------------------

checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "sft_model.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)

config = checkpoint["config"]


# ------------------------------------------------------------
# Rebuild tokenizer
# ------------------------------------------------------------

tokenizer = BPETokenizer(
    vocab_size=config["bpe_vocab_size"]
)

tokenizer.load_merges(
    checkpoint["tokenizer_merges"]
)


# ------------------------------------------------------------
# Rebuild model
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Prompt
# ------------------------------------------------------------

prompt = (
    "User: What is 1 + 1?\n"
    "Assistant:"
)

prompt_ids = tokenizer.encode(prompt)

context = torch.tensor(
    [prompt_ids],
    dtype=torch.long,
    device=device
)


# ------------------------------------------------------------
# Generate
# ------------------------------------------------------------

def generate_response(model, tokenizer, prompt, config, device, max_new_tokens=30):
    prompt_ids = tokenizer.encode(prompt)
    context = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context_for_model = context[:, -config["max_sequence_length"]:]
            logits = model(context_for_model)

            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            if next_token.item() == tokenizer.eos_id:
                break

            context = torch.cat([context, next_token], dim=1)

    return tokenizer.decode(context[0].tolist())

test_prompts = [
    "User: What is 1 + 1?\nAssistant:",
    "User: What is 4 + 7?\nAssistant:",
    "User: What is 7 + 8?\nAssistant:",
    "User: What is 2 + 9?\nAssistant:",
    "User: What is 6 + 6?\nAssistant:",
]


for prompt in test_prompts:

    output = generate_response(
        model,
        tokenizer,
        prompt,
        config,
        device
    )

    print()
    print(output)
    print("-" * 40)
