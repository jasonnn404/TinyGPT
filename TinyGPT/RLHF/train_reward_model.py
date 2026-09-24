import torch
import torch.nn.functional as F

from model import TinyGPT
from bpe_tokenizer import BPETokenizer
from RLHF.reward_model import RewardModel


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)


# ------------------------------------------------------------
# Load SFT checkpoint
# ------------------------------------------------------------

checkpoint = torch.load("checkpoints/sft_model.pt", map_location=device)
config = checkpoint["config"]


# ------------------------------------------------------------
# Restore tokenizer
# ------------------------------------------------------------

tokenizer = BPETokenizer(vocab_size=config["bpe_vocab_size"])
tokenizer.load_merges(checkpoint["tokenizer_merges"])

print("Model vocab size:", config["vocab_size"])
print("PAD ID:", tokenizer.pad_id)
print("EOS ID:", tokenizer.eos_id)


# ------------------------------------------------------------
# Restore SFT backbone
# ------------------------------------------------------------

backbone = TinyGPT(
    vocab_size=config["vocab_size"],
    max_sequence_length=config["max_sequence_length"],
    embedding_dimension=config["embedding_dimension"],
    num_heads=config["num_heads"],
    num_layers=config["num_layers"],
    dropout=config["dropout"]
).to(device)

backbone.load_state_dict(checkpoint["model_state_dict"])

print("Loaded SFT backbone.")


# ------------------------------------------------------------
# Build Reward Model
# ------------------------------------------------------------

reward_model = RewardModel(backbone).to(device)


# ------------------------------------------------------------
# Preference data
# ------------------------------------------------------------

preference_data = []

for a in range(10):
    for b in range(10):
        correct = a + b

        prompt = f"User: What is {a} + {b}?\nAssistant:"
        chosen = f" {a} + {b} is {correct}."

        rejected_responses = set()

        # ----------------------------------------------------
        # 1. Wrong answer, same operands
        # ----------------------------------------------------

        for offset in [-3, -2, -1, 1, 2, 3]:
            wrong = correct + offset

            if wrong >= 0:
                rejected_responses.add(
                    f" {a} + {b} is {wrong}."
                )

        # ----------------------------------------------------
        # 2. Correct arithmetic, WRONG operands
        # ----------------------------------------------------

        other_a = (a + 1) % 10
        other_b = (b + 1) % 10

        rejected_responses.add(
            f" {other_a} + {b} is {other_a + b}."
        )

        rejected_responses.add(
            f" {a} + {other_b} is {a + other_b}."
        )

        rejected_responses.add(
            f" {other_a} + {other_b} is {other_a + other_b}."
        )

        # ----------------------------------------------------
        # 3. Malformed / distracting responses
        # ----------------------------------------------------

        rejected_responses.add(
            f" {a} + {b} is {correct}. 0 + 0 is 0."
        )

        rejected_responses.add(
            f" {a} + {b} is {correct} is {correct + 1}."
        )

        # ----------------------------------------------------
        # Add preference pairs
        # ----------------------------------------------------

        for rejected in rejected_responses:
            preference_data.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            })

print("Preference pairs:", len(preference_data))

# ------------------------------------------------------------
# Encode prompt + response + EOS
# ------------------------------------------------------------

def encode_example(prompt, response):
    ids = tokenizer.encode(prompt + response)
    ids.append(tokenizer.eos_id)
    return ids


chosen_sequences = []
rejected_sequences = []

for example in preference_data:
    chosen_sequences.append(encode_example(example["prompt"], example["chosen"]))
    rejected_sequences.append(encode_example(example["prompt"], example["rejected"]))


# ------------------------------------------------------------
# Padding
# ------------------------------------------------------------

def pad_sequences(sequences, max_length, pad_id):
    padded = []
    masks = []

    for sequence in sequences:
        sequence = sequence[:max_length]
        padding_length = max_length - len(sequence)

        padded.append(sequence + [pad_id] * padding_length)
        masks.append([1] * len(sequence) + [0] * padding_length)

    return (
        torch.tensor(padded, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device)
    )


longest_sequence = max(
    max(len(seq) for seq in chosen_sequences),
    max(len(seq) for seq in rejected_sequences)
)

batch_sequence_length = min(longest_sequence, config["max_sequence_length"])

chosen_ids, chosen_mask = pad_sequences(
    chosen_sequences, batch_sequence_length, tokenizer.pad_id
)

rejected_ids, rejected_mask = pad_sequences(
    rejected_sequences, batch_sequence_length, tokenizer.pad_id
)

print("Chosen shape:", chosen_ids.shape)
print("Rejected shape:", rejected_ids.shape)


# ------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------

optimizer = torch.optim.AdamW(reward_model.parameters(), lr=1e-4)
num_steps = 200


# ------------------------------------------------------------
# Train Reward Model
# ------------------------------------------------------------

for step in range(num_steps):
    reward_model.train()

    chosen_reward = reward_model(chosen_ids, chosen_mask)
    rejected_reward = reward_model(rejected_ids, rejected_mask)

    reward_difference = chosen_reward - rejected_reward

    # Pairwise preference loss
    loss = -F.logsigmoid(reward_difference).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        with torch.no_grad():
            accuracy = (chosen_reward > rejected_reward).float().mean()

        print(
            f"step {step:3d} | "
            f"loss {loss.item():.4f} | "
            f"chosen {chosen_reward.mean().item():.4f} | "
            f"rejected {rejected_reward.mean().item():.4f} | "
            f"diff {reward_difference.mean().item():.4f} | "
            f"accuracy {accuracy.item():.2f}"
        )


# ------------------------------------------------------------
# Save Reward Model
# ------------------------------------------------------------

torch.save(
    {
        "reward_model_state_dict": reward_model.state_dict(),
        "config": config,
        "tokenizer_merges": checkpoint["tokenizer_merges"]
    },
    "checkpoints/reward_model.pt"
)

print("Saved Reward Model to checkpoints/reward_model.pt")