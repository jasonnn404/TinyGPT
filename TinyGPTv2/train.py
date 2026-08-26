import torch
import torch.nn.functional as F

from model import TinyGPT
from bpe_tokenizer import BPETokenizer
from data import TextData


# -------------------------
# Training configuration
# -------------------------

resume_training = False

batch_size = 4
block_size = 64

max_steps = 10
eval_interval = 5
eval_iters = 20
save_interval = 1000

learning_rate = 0.001

bpe_vocab_size = 500


# -------------------------
# Load text
# -------------------------

with open("data.txt", "r") as f:
    text = f.read()


# -------------------------
# Load checkpoint
# -------------------------

checkpoint = None

if resume_training:
    checkpoint = torch.load(
        "latest_checkpoint.pt"
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
        "embedding_dimension": 64,
        "num_heads": 4,
        "num_layers": 2
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


# -------------------------
# Optimizer
# -------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
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


    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate

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
                "tokenizer_vocab_size": tokenizer.vocab_size,
                "merges": tokenizer.merges,
                "model_config": model_config,
            }, "best_model.pt")

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
            "learning_rate": learning_rate,
        }, "latest_checkpoint.pt")


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