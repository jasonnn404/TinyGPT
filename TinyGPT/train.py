import torch
import torch.nn.functional as F

from .model import TinyGPT
from .bpe_tokenizer import BPETokenizer
from .data import TextData

from pathlib import Path
import math


# -------------------------
# Training configuration
# -------------------------

resume_training = False

batch_size = 4
block_size = 128


max_steps = 10000
warmup_steps = 500
eval_interval = 500
eval_iters = 20
save_interval = 1000


max_lr = 0.001
min_lr = 0.0001


bpe_vocab_size = 500


# -------------------------
# Learning rate schedule
# -------------------------


def get_lr(step):

    # 1. Warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # 2. Training finished
    if step >= max_steps:
        return min_lr

    # 3. Cosine decay
    decay_ratio = (
        (step - warmup_steps)
        / (max_steps - warmup_steps)
    )

    coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )

    return min_lr + coeff * (max_lr - min_lr)


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
# Load checkpoint
# -------------------------

checkpoint = None

if resume_training:
    checkpoint = torch.load(
        latest_checkpoint_path,
        map_location="cpu"
    )


# -------------------------
# Tokenizer
# -------------------------

if resume_training:
    tokenizer = BPETokenizer(
        vocab_size=checkpoint["tokenizer_vocab_size"]
    )

    tokenizer.load_merges(
        checkpoint["merges"]
    )

else:
    tokenizer = BPETokenizer(
        vocab_size=bpe_vocab_size
    )

    tokenizer.train(text)


# -------------------------
# Model configuration
# -------------------------

if resume_training:
    model_config = checkpoint["model_config"]

    block_size = model_config["max_sequence_length"]

else:
    model_config = {
        "vocab_size": tokenizer.vocab_size,
        "max_sequence_length": block_size,
        "embedding_dimension": 128,
        "num_heads": 4,
        "num_layers": 4,
        "dropout": 0.1
    }


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

num_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Number of parameters: {num_params / 1e6:.2f}M"
)

# -------------------------
# Optimizer
# -------------------------

decay_params = []
no_decay_params = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue

    if param.dim() >= 2:
        decay_params.append(param)
    else:
        no_decay_params.append(param)


optimizer = torch.optim.AdamW(
    [
        {
            "params": decay_params,
            "weight_decay": 0.01
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0
        }
    ],
    lr=max_lr
)


# -------------------------
# Resume training state
# -------------------------

start_step = 0
best_val_loss = float("inf")

if resume_training:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_step = checkpoint["step"] + 1
    best_val_loss = checkpoint["best_val_loss"]

    print("Resuming from step:", start_step)


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
# Training loop
# -------------------------

for step in range(start_step, max_steps):

    lr = get_lr(step)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # Evaluation
    if step % eval_interval == 0:

        losses = estimate_loss()

        print(
            "step:", step,
            "lr:", lr,
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
                "tokenizer_vocab_size": tokenizer.vocab_size,
                "merges": tokenizer.merges,
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
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "merges": tokenizer.merges,
            "model_config": model_config,
            "batch_size": batch_size,
            "current_learning_rate": lr,
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

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=2.0
    )

    optimizer.step()

    if step % eval_interval == 0:
        print("gradient norm:", grad_norm.item())
