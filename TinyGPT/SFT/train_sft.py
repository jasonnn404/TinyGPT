import os

import torch
import torch.nn.functional as F

from model import TinyGPT
from bpe_tokenizer import BPETokenizer


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

bpe_vocab_size = 500
max_sequence_length = 128

embedding_dimension = 128
num_heads = 4
num_layers = 4
dropout = 0.1

learning_rate = 3e-4
num_steps = 500


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)


# ------------------------------------------------------------
# SFT data
# ------------------------------------------------------------

# ------------------------------------------------------------
# SFT data
# ------------------------------------------------------------

held_out_pairs = {
    (1, 1),
    (4, 7),
    (7, 8),
    (2, 9),
    (6, 6),
}


sft_data = []
validation_data = []


for a in range(10):

    for b in range(10):

        example = {
            "prompt": (
                f"User: What is {a} + {b}?\n"
                "Assistant:"
            ),
            "response": (
                f" {a} + {b} is {a + b}."
            )
        }

        if (a, b) in held_out_pairs:

            validation_data.append(
                example
            )

        else:

            sft_data.append(
                example
            )


# Add a few non-math instructions
sft_data += [
    {
        "prompt": (
            "User: What is the capital of France?\n"
            "Assistant:"
        ),
        "response": (
            " The capital of France is Paris."
        )
    },

    {
        "prompt": (
            "User: What planet do humans live on?\n"
            "Assistant:"
        ),
        "response": (
            " Humans live on Earth."
        )
    },

    {
        "prompt": (
            "User: Say hello.\n"
            "Assistant:"
        ),
        "response": " Hello!"
    },

    {
        "prompt": (
            "User: What color is the sky?\n"
            "Assistant:"
        ),
        "response": " The sky is blue."
    },
]


# ------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------

tokenizer = BPETokenizer(
    vocab_size=bpe_vocab_size
)

model_vocab_size = tokenizer.model_vocab_size

# Train the BPE tokenizer on the SFT text.
#
# Later, if we start from a real pretrained checkpoint,
# we should load that model's existing tokenizer instead.

tokenizer_training_text = ""

for example in sft_data:
    tokenizer_training_text += (
        example["prompt"]
        + example["response"]
        + "\n"
    )

tokenizer.train(
    tokenizer_training_text
)


# ------------------------------------------------------------
# Encode SFT examples
# ------------------------------------------------------------

input_sequences = []
label_sequences = []

for example in sft_data:

    prompt_ids = tokenizer.encode(
        example["prompt"]
    )

    response_ids = tokenizer.encode(
        example["response"]
    )

    response_ids.append(
        tokenizer.eos_id
    )

    # Model sees BOTH prompt and response
    input_ids = (
        prompt_ids
        + response_ids
    )

    # But loss is only calculated on response tokens
    labels = (
        [-100] * len(prompt_ids)
        + response_ids
    )

    input_sequences.append(
        input_ids
    )

    label_sequences.append(
        labels
    )


# ------------------------------------------------------------
# Padding
# ------------------------------------------------------------

longest_sequence = max(
    len(sequence)
    for sequence in input_sequences
)

batch_sequence_length = min(
    longest_sequence,
    max_sequence_length
)


def pad_example(input_ids, labels, max_length, pad_id):

    # Truncate if sequence is too long
    input_ids = input_ids[:max_length]
    labels = labels[:max_length]

    padding_length = (
        max_length - len(input_ids)
    )

    # Padding token IDs
    input_ids = (
        input_ids
        + [pad_id] * padding_length
    )

    # Padding should NOT contribute to loss
    labels = (
        labels
        + [-100] * padding_length
    )

    return input_ids, labels


padded_inputs = []
padded_labels = []

for input_ids, labels in zip(
    input_sequences,
    label_sequences
):

    input_ids, labels = pad_example(
        input_ids,
        labels,
        batch_sequence_length,
        tokenizer.pad_id
    )

    padded_inputs.append(
        input_ids
    )

    padded_labels.append(
        labels
    )


# ------------------------------------------------------------
# Convert to tensors
# ------------------------------------------------------------

input_ids = torch.tensor(
    padded_inputs,
    dtype=torch.long,
    device=device
)

labels = torch.tensor(
    padded_labels,
    dtype=torch.long,
    device=device
)


print(
    "Input shape:",
    input_ids.shape
)

print(
    "Label shape:",
    labels.shape
)


# ------------------------------------------------------------
# Build TinyGPT
# ------------------------------------------------------------

model = TinyGPT(
    vocab_size=model_vocab_size,
    max_sequence_length=max_sequence_length,
    embedding_dimension=embedding_dimension,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout=dropout
).to(device)


# ------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# ------------------------------------------------------------
# SFT training
# ------------------------------------------------------------

for step in range(num_steps):

    model.train()

    # [B, T] -> [B, T, V]
    logits = model(input_ids)

    # --------------------------------------------------------
    # Shift for next-token prediction
    # --------------------------------------------------------

    shift_logits = logits[:, :-1, :]
    # [B, T-1, V]

    shift_labels = labels[:, 1:]
    # [B, T-1]

    # --------------------------------------------------------
    # Cross entropy
    #
    # -100 positions are ignored:
    #
    # prompt tokens -> ignored
    # padding       -> ignored
    # response      -> trained
    # --------------------------------------------------------

    loss = F.cross_entropy(
        shift_logits.reshape(
            -1,
            model_vocab_size
        ),
        shift_labels.reshape(-1),
        ignore_index=-100
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    if step % 50 == 0:
        print(
            f"step {step:3d} | "
            f"loss {loss.item():.4f}"
        )


# ------------------------------------------------------------
# Save checkpoint
# ------------------------------------------------------------

os.makedirs(
    "checkpoints",
    exist_ok=True
)

checkpoint = {
    "model_state_dict": model.state_dict(),

    "config": {
        "bpe_vocab_size": bpe_vocab_size,
        "vocab_size": model_vocab_size,
        "max_sequence_length": max_sequence_length,
        "embedding_dimension": embedding_dimension,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "dropout": dropout,
    },

    # IMPORTANT:
    # Reward Model / PPO must use the exact same BPE merges
    "tokenizer_merges": tokenizer.merges,
}

torch.save(
    checkpoint,
    "checkpoints/sft_model.pt"
)

print(
    "Saved SFT checkpoint to "
    "checkpoints/sft_model.pt"
)