import torch

from model import TinyGPT
from bpe_tokenizer import BPETokenizer
from RLHF.reward_model import RewardModel


device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

sft_checkpoint = torch.load("checkpoints/sft_model.pt", map_location=device)
reward_checkpoint = torch.load("checkpoints/reward_model.pt", map_location=device)

config = sft_checkpoint["config"]

tokenizer = BPETokenizer(vocab_size=config["bpe_vocab_size"])
tokenizer.load_merges(sft_checkpoint["tokenizer_merges"])


def build_sft_model():
    model = TinyGPT(
        vocab_size=config["vocab_size"],
        max_sequence_length=config["max_sequence_length"],
        embedding_dimension=config["embedding_dimension"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"]
    ).to(device)

    model.load_state_dict(sft_checkpoint["model_state_dict"])
    return model


backbone = build_sft_model()

reward_model = RewardModel(backbone).to(device)
reward_model.load_state_dict(reward_checkpoint["reward_model_state_dict"])
reward_model.eval()


def score_response(prompt, response):
    ids = tokenizer.encode(prompt + response)
    ids.append(tokenizer.eos_id)

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        reward = reward_model(input_ids, attention_mask)

    return reward.item()


# ------------------------------------------------------------
# Evaluate all addition preference pairs
# ------------------------------------------------------------

correct_preferences = 0
total = 0
mistakes = []

for a in range(10):
    for b in range(10):
        correct = a + b
        wrong = correct + 1 if correct < 18 else correct - 1

        prompt = f"User: What is {a} + {b}?\nAssistant:"

        chosen = f" {a} + {b} is {correct}."
        rejected = f" {a} + {b} is {wrong}."

        chosen_reward = score_response(prompt, chosen)
        rejected_reward = score_response(prompt, rejected)

        total += 1

        if chosen_reward > rejected_reward:
            correct_preferences += 1
        else:
            mistakes.append({
                "question": f"{a} + {b}",
                "chosen_reward": chosen_reward,
                "rejected_reward": rejected_reward,
                "correct": correct,
                "wrong": wrong
            })


accuracy = correct_preferences / total

print("\nReward Model Preference Accuracy")
print("-" * 40)
print(f"Correct preferences: {correct_preferences}/{total}")
print(f"Accuracy: {accuracy * 100:.1f}%")

print("\nMistakes")
print("-" * 40)

for mistake in mistakes[:20]:
    print(
        f"{mistake['question']} | "
        f"correct={mistake['correct']} score={mistake['chosen_reward']:.4f} | "
        f"wrong={mistake['wrong']} score={mistake['rejected_reward']:.4f}"
    )


# ------------------------------------------------------------
# Explicitly inspect the 5 held-out SFT questions
# ------------------------------------------------------------
current_rlhf_errors = {
    (1, 1): 3,
    (2, 9): 14,
    (4, 7): 10,
    (6, 6): 15
}

print("\nCurrent RLHF mistakes")
print("-" * 40)

for (a, b), wrong in current_rlhf_errors.items():
    correct = a + b

    prompt = f"User: What is {a} + {b}?\nAssistant:"

    correct_response = f" {a} + {b} is {correct}."
    wrong_response = f" {a} + {b} is {wrong}."

    correct_score = score_response(prompt, correct_response)
    wrong_score = score_response(prompt, wrong_response)

    print(
        f"{a} + {b} | "
        f"correct {correct}: {correct_score:.4f} | "
        f"wrong {wrong}: {wrong_score:.4f} | "
        f"margin: {correct_score - wrong_score:+.4f}"
    )

off_prompt_tests = [
    (
        "User: What is 4 + 7?\nAssistant:",
        " 4 + 7 is 11.",
        " 4 + 9 is 13."
    ),
    (
        "User: What is 6 + 6?\nAssistant:",
        " 6 + 6 is 12.",
        " 6 + 7 is 13."
    ),
    (
        "User: What is 7 + 8?\nAssistant:",
        " 7 + 8 is 15.",
        " 7 + 6 is 13."
    )
]

print("\nOff-prompt tests")
print("-" * 40)

for prompt, correct_response, wrong_response in off_prompt_tests:
    correct_score = score_response(prompt, correct_response)
    wrong_score = score_response(prompt, wrong_response)

    print(prompt.splitlines()[0])
    print(f"correct: {correct_score:.4f}")
    print(f"off-prompt: {wrong_score:.4f}")
    print(f"margin: {correct_score - wrong_score:+.4f}")
    print()