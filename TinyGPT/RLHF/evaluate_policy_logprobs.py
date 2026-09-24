import torch
import torch.nn.functional as F

from model import TinyGPT
from bpe_tokenizer import BPETokenizer


device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

sft_checkpoint = torch.load("checkpoints/sft_model.pt", map_location=device)
rlhf_checkpoint = torch.load("checkpoints/rlhf_model.pt", map_location=device)

config = sft_checkpoint["config"]

tokenizer = BPETokenizer(vocab_size=config["bpe_vocab_size"])
tokenizer.load_merges(sft_checkpoint["tokenizer_merges"])


def build_model():
    return TinyGPT(
        vocab_size=config["vocab_size"],
        max_sequence_length=config["max_sequence_length"],
        embedding_dimension=config["embedding_dimension"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"]
    ).to(device)


sft_policy = build_model()
sft_policy.load_state_dict(sft_checkpoint["model_state_dict"])
sft_policy.eval()

rlhf_policy = build_model()
rlhf_policy.load_state_dict(rlhf_checkpoint["policy_state_dict"])
rlhf_policy.eval()


def response_logprob(model, prompt, response):
    prompt_ids = tokenizer.encode(prompt)
    response_ids = tokenizer.encode(response) + [tokenizer.eos_id]

    full_ids = prompt_ids + response_ids

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_ids)

    prompt_length = len(prompt_ids)
    response_length = len(response_ids)

    action_logits = logits[
        :,
        prompt_length - 1 : prompt_length - 1 + response_length,
        :
    ]

    actions = input_ids[
        :,
        prompt_length : prompt_length + response_length
    ]

    log_probs = F.log_softmax(action_logits, dim=-1)

    token_logprobs = log_probs.gather(
        -1,
        actions.unsqueeze(-1)
    ).squeeze(-1)

    return token_logprobs.sum().item()


cases = {
    (1, 1): 3,
    (2, 9): 14,
    (4, 7): 10,
    (6, 6): 15,
    (7, 8): 16
}


for (a, b), wrong in cases.items():
    correct = a + b

    prompt = f"User: What is {a} + {b}?\nAssistant:"

    correct_response = f" {a} + {b} is {correct}."
    wrong_response = f" {a} + {b} is {wrong}."

    sft_correct = response_logprob(sft_policy, prompt, correct_response)
    sft_wrong = response_logprob(sft_policy, prompt, wrong_response)

    rlhf_correct = response_logprob(rlhf_policy, prompt, correct_response)
    rlhf_wrong = response_logprob(rlhf_policy, prompt, wrong_response)

    print(f"\n{a} + {b}")

    print(
        f"SFT  | correct {sft_correct:.4f} | "
        f"wrong {sft_wrong:.4f} | "
        f"margin {sft_correct - sft_wrong:+.4f}"
    )

    print(
        f"RLHF | correct {rlhf_correct:.4f} | "
        f"wrong {rlhf_wrong:.4f} | "
        f"margin {rlhf_correct - rlhf_wrong:+.4f}"
    )