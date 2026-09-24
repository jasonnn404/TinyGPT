import torch
import torch.nn.functional as F
from pathlib import Path

from .model import TinyGPT
from .tokenizer import CharTokenizer
from .data import TextData


# -------------------------
# Load text
# -------------------------

project_dir = Path(__file__).parent
data_path = project_dir / "data.txt"
latest_checkpoint_path = project_dir / "latest_checkpoint.pt"
best_model_path = project_dir / "best_model.pt"

with open(data_path, "r") as f:
    text = f.read()


# -------------------------
# Tokenizer
# -------------------------

tokenizer = CharTokenizer(text=text)


# -------------------------
# Training configuration
# -------------------------

batch_size = 4
block_size = 64

max_steps = 10000

eval_interval = 500
eval_iters = 20

save_interval = 1000

learning_rate = 0.001


# -------------------------
# Dataset
# -------------------------

dataset = TextData(
    text=text,
    tokenizer=tokenizer,
    batch_size=batch_size,
    block_size=block_size
)


# -------------------------
# Model configuration
# -------------------------

model_config = {
    "vocab_size": len(tokenizer.characters),
    "max_sequence_length": block_size,
    "embedding_dimension": 64,
    "num_heads": 4,
    "num_layers": 2
}


# -------------------------
# Model
# -------------------------

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)

model = TinyGPT(
    **model_config
).to(device)


# -------------------------
# Optimizer
# -------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# -------------------------
# Estimate loss
# -------------------------

@torch.no_grad()
def estimate_loss():

    out = {}

    model.eval()

    for split in ["train", "val"]:

        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):

            X, Y = dataset.get_batch(split)

            X = X.to(device)
            Y = Y.to(device)

            logits = model(X)

            B, T, C = logits.shape

            logits = logits.reshape(B * T, C)
            Y = Y.reshape(B * T)

            loss = F.cross_entropy(
                logits,
                Y
            )

            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()

    return out


# -------------------------
# Resume training
# -------------------------

resume_training = False

start_step = 0
best_val_loss = float("inf")


if resume_training:

    checkpoint = torch.load(
        latest_checkpoint_path,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_step = checkpoint["step"] + 1

    best_val_loss = checkpoint["best_val_loss"]

    print(
        "Resuming from step:",
        start_step
    )


# -------------------------
# Training loop
# -------------------------

for step in range(start_step, max_steps):

    # Evaluation
    if step % eval_interval == 0:

        losses = estimate_loss()

        print(
            "step:", step,
            "train loss:", losses["train"],
            "val loss:", losses["val"]
        )

        val_loss = losses["val"].item()

        # Save best model
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "best_val_loss": best_val_loss,
                "characters": tokenizer.characters,
                "model_config": model_config,
            }, best_model_path)

            print("saved best model")


    # Save latest checkpoint
    if step % save_interval == 0:

        torch.save({
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "characters": tokenizer.characters,
            "model_config": model_config,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        }, latest_checkpoint_path)


    # Get training batch
    x, y = dataset.get_batch("train")

    x = x.to(device)
    y = y.to(device)

    # Forward
    logits = model(x)

    B, T, C = logits.shape

    logits = logits.reshape(B * T, C)
    y = y.reshape(B * T)


    # Loss
    loss = F.cross_entropy(
        logits,
        y
    )


    # Backpropagation
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
