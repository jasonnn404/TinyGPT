import torch
import torch.nn.functional as F

from generation import generate
from model import TinyGPT
from tokenizer import CharTokenizer


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

checkpoint = torch.load(
    "best_model.pt"
)


tokenizer = CharTokenizer(
    characters=checkpoint["characters"]
)


model = TinyGPT(
    **checkpoint["model_config"]
).to(device)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()



block_size = checkpoint["model_config"]["max_sequence_length"]

generated_text = generate(
    model=model,
    tokenizer=tokenizer,
    prompt="hello",
    block_size=block_size,
    max_new_tokens=100,
    temperature=0.8,
    k=5
)

print(generated_text)