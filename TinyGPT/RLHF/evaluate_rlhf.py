import re
import torch

from model import TinyGPT
from bpe_tokenizer import BPETokenizer


# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)


# ------------------------------------------------------------
# Load SFT checkpoint
# ------------------------------------------------------------

sft_checkpoint = torch.load("checkpoints/sft_model.pt", map_location=device)
config = sft_checkpoint["config"]

tokenizer = BPETokenizer(vocab_size=config["bpe_vocab_size"])
tokenizer.load_merges(sft_checkpoint["tokenizer_merges"])


def build_model():
    model = TinyGPT(
        vocab_size=config["vocab_size"],
        max_sequence_length=config["max_sequence_length"],
        embedding_dimension=config["embedding_dimension"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"]
    ).to(device)

    return model


# ------------------------------------------------------------
# SFT Policy
# ------------------------------------------------------------

sft_policy = build_model()
sft_policy.load_state_dict(sft_checkpoint["model_state_dict"])
sft_policy.eval()


# ------------------------------------------------------------
# RLHF Policy
# ------------------------------------------------------------

rlhf_checkpoint = torch.load("checkpoints/rlhf_model.pt", map_location=device)

rlhf_policy = build_model()
rlhf_policy.load_state_dict(rlhf_checkpoint["policy_state_dict"])
rlhf_policy.eval()


# ------------------------------------------------------------
# Greedy generation
# ------------------------------------------------------------

def generate_response(model, prompt, max_new_tokens=30):
    prompt_ids = tokenizer.encode(prompt)

    context = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device
    )

    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if context.size(1) >= config["max_sequence_length"]:
                break

            logits = model(context)
            next_token_logits = logits[:, -1, :]

            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True
            )

            token_id = next_token.item()

            if token_id == tokenizer.eos_id:
                break

            generated_tokens.append(token_id)

            context = torch.cat(
                [context, next_token],
                dim=1
            )

    return tokenizer.decode(generated_tokens)


# ------------------------------------------------------------
# Extract predicted answer
# ------------------------------------------------------------

def extract_answer(response):
    numbers = re.findall(r"-?\d+", response)

    if not numbers:
        return None

    return int(numbers[-1])


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def evaluate_model(model, name):
    correct = 0
    total = 0
    mistakes = []

    for a in range(10):
        for b in range(10):
            prompt = f"User: What is {a} + {b}?\nAssistant:"
            target = a + b

            response = generate_response(model, prompt)
            prediction = extract_answer(response)

            total += 1

            if prediction == target:
                correct += 1
            else:
                mistakes.append({
                    "prompt": prompt,
                    "response": response,
                    "prediction": prediction,
                    "target": target
                })

    accuracy = correct / total

    print(f"\n{name}")
    print("-" * 40)
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy * 100:.1f}%")

    return accuracy, mistakes


# ------------------------------------------------------------
# Compare SFT vs RLHF
# ------------------------------------------------------------

sft_accuracy, sft_mistakes = evaluate_model(
    sft_policy,
    "SFT Policy"
)

rlhf_accuracy, rlhf_mistakes = evaluate_model(
    rlhf_policy,
    "RLHF Policy"
)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\nComparison")
print("-" * 40)
print(f"SFT accuracy:  {sft_accuracy * 100:.1f}%")
print(f"RLHF accuracy: {rlhf_accuracy * 100:.1f}%")
print(
    f"Change:        "
    f"{(rlhf_accuracy - sft_accuracy) * 100:+.1f}%"
)


# ------------------------------------------------------------
# Show some RLHF mistakes
# ------------------------------------------------------------

print("\nSFT mistakes:")
for mistake in sft_mistakes:
    print(
        mistake["prompt"].splitlines()[0],
        "| response:", mistake["response"],
        "| target:", mistake["target"]
    )

print("\nRLHF mistakes:")
for mistake in rlhf_mistakes:
    print(
        mistake["prompt"].splitlines()[0],
        "| response:", mistake["response"],
        "| target:", mistake["target"]
    )