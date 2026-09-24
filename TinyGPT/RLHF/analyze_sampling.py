import re
from collections import Counter, defaultdict

import torch

from model import TinyGPT
from bpe_tokenizer import BPETokenizer
from RLHF.reward_model import RewardModel


# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

sft_checkpoint = torch.load("checkpoints/sft_model.pt", map_location=device)
rlhf_checkpoint = torch.load("checkpoints/rlhf_model.pt", map_location=device)
reward_checkpoint = torch.load("checkpoints/reward_model.pt", map_location=device)

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


# ------------------------------------------------------------
# SFT Policy
# ------------------------------------------------------------

sft_policy = build_model()
sft_policy.load_state_dict(sft_checkpoint["model_state_dict"])
sft_policy.eval()


# ------------------------------------------------------------
# RLHF Policy
# ------------------------------------------------------------

rlhf_policy = build_model()
rlhf_policy.load_state_dict(rlhf_checkpoint["policy_state_dict"])
rlhf_policy.eval()


# ------------------------------------------------------------
# Reward Model
# ------------------------------------------------------------

reward_backbone = build_model()

reward_model = RewardModel(reward_backbone).to(device)
reward_model.load_state_dict(reward_checkpoint["reward_model_state_dict"])
reward_model.eval()


# ------------------------------------------------------------
# Sample one response
# ------------------------------------------------------------

def sample_response(model, prompt, max_new_tokens=30):
    prompt_ids = tokenizer.encode(prompt)
    context = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if context.size(1) >= config["max_sequence_length"]:
                break

            logits = model(context)
            next_token_logits = logits[:, -1, :]

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            token_id = next_token.item()

            if token_id == tokenizer.eos_id:
                break

            generated_tokens.append(token_id)
            context = torch.cat([context, next_token], dim=1)

    return tokenizer.decode(generated_tokens)


# ------------------------------------------------------------
# Extract final numeric answer
# ------------------------------------------------------------

def extract_answer(response):
    numbers = re.findall(r"-?\d+", response)

    if not numbers:
        return None

    return int(numbers[-1])


# ------------------------------------------------------------
# Reward Model score
# ------------------------------------------------------------

def score_response(prompt, response):
    ids = tokenizer.encode(prompt + response)
    ids.append(tokenizer.eos_id)

    if len(ids) > config["max_sequence_length"]:
        ids = ids[:config["max_sequence_length"]]

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        reward = reward_model(input_ids, attention_mask)

    return reward.item()


# ------------------------------------------------------------
# Analyze one policy on one question
# ------------------------------------------------------------

def analyze_question(model, model_name, a, b, num_samples=100):
    prompt = f"User: What is {a} + {b}?\nAssistant:"
    correct_answer = a + b

    answer_counts = Counter()
    responses_by_answer = defaultdict(list)
    rewards_by_answer = defaultdict(list)

    for _ in range(num_samples):
        response = sample_response(model, prompt)
        answer = extract_answer(response)
        reward = score_response(prompt, response)

        answer_counts[answer] += 1
        responses_by_answer[answer].append(response)
        rewards_by_answer[answer].append(reward)

    print(f"\n{'=' * 60}")
    print(f"{model_name} | {a} + {b} = {correct_answer}")
    print(f"{'=' * 60}")

    sorted_answers = answer_counts.most_common()

    for answer, count in sorted_answers:
        rewards = rewards_by_answer[answer]
        mean_reward = sum(rewards) / len(rewards)

        marker = "✅" if answer == correct_answer else "❌"

        print(
            f"{marker} answer={answer} | "
            f"count={count:3d}/{num_samples} | "
            f"mean RM reward={mean_reward:.4f}"
        )

        example = responses_by_answer[answer][0]
        print(f"   example: {example}")

    correct_count = answer_counts[correct_answer]

    print(
        f"\nCorrect sampling rate: "
        f"{correct_count}/{num_samples} "
        f"({correct_count / num_samples * 100:.1f}%)"
    )


# ------------------------------------------------------------
# Cases
# ------------------------------------------------------------

cases = [
    (1, 1),
    (2, 9),
    (4, 7),
    (6, 6),
    (7, 8)
]


# ------------------------------------------------------------
# Run analysis
# ------------------------------------------------------------

num_samples = 100

for a, b in cases:
    analyze_question(
        sft_policy,
        "SFT",
        a,
        b,
        num_samples=num_samples
    )

    analyze_question(
        rlhf_policy,
        "RLHF",
        a,
        b,
        num_samples=num_samples
    )