import torch

from .generation import generate_greedy_cached
from .model import TinyGPT
from .bpe_tokenizer import BPETokenizer

from pathlib import Path

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

checkpoint_path = Path(__file__).parent / "best_model.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)


tokenizer = BPETokenizer(
    vocab_size=checkpoint["tokenizer_vocab_size"]
)

tokenizer.load_merges(
    checkpoint["merges"]
)


model = TinyGPT(
    **checkpoint["model_config"]
).to(device)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()



prompt_ids = tokenizer.encode("hello")
input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
max_new_tokens = min(
    50,
    checkpoint["model_config"]["max_sequence_length"] - len(prompt_ids),
)
generated_ids = generate_greedy_cached(model, input_ids, max_new_tokens)
generated_text = tokenizer.decode(generated_ids[0].tolist())

print(generated_text)
