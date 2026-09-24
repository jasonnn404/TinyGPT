# TinyGPT

A from-scratch GPT-style decoder-only transformer project. The current implementation includes byte-pair encoding, KV-cache generation, supervised fine-tuning, and RLHF experiments, while the original character-level implementation is retained as `TinyGPTv1`.

## Project snapshots

| Folder | Tokenizer | Focus |
| --- | --- | --- |
| `TinyGPTv1/` | Character-level | The original implementation of causal self-attention, multi-head attention, transformer blocks, training, checkpointing, and generation |
| `TinyGPT/` | Byte-pair encoding (BPE) | The current implementation, including BPE tokenization, KV-cache generation, supervised fine-tuning, and RLHF experiments |

Both snapshots currently use a two-layer model with a 64-dimensional embedding, four attention heads, and a context length of 64. Small trained checkpoints are included so generation can be tried immediately. The training corpus in both folders is the same text.

## Set up on another computer

Python 3.10 or newer is recommended. PyTorch installation can differ for CPU, CUDA, and other accelerators; use the command from the [PyTorch installation selector](https://pytorch.org/get-started/locally/) if the simple installation below does not match your PC.

```bash
git clone https://github.com/jasonnn404/TinyGPT.git
cd TinyGPT
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the models

Run the scripts as modules from the repository root.

Character model:

```bash
python -m TinyGPTv1.generate
python -m TinyGPTv1.train
```

BPE model:

```bash
python -m TinyGPT.generate
python -m TinyGPT.train
```

Training overwrites `best_model.pt` when validation loss improves and periodically writes `latest_checkpoint.pt`. Set `resume_training = True` in the relevant `train.py` to resume from the latest checkpoint. Before a new experiment, copy any checkpoint you want to preserve or record its Git commit.

To compare character and BPE token counts, run this from the repository root:

```bash
python -m TinyGPT.tests.test_tokenizer
```

## Documenting experiments

- Add learning milestones to [`docs/study-log.md`](docs/study-log.md).
- Copy [`experiments/TEMPLATE.md`](experiments/TEMPLATE.md) for each meaningful run.
- Record the code commit, configuration, device, loss, runtime, and a sample output.
- Commit source and notes frequently. Avoid committing large generated checkpoints or secrets.

## Current limitations and next questions

- Training configuration is edited directly in the scripts rather than passed as command-line arguments.
- Random seeds are not fixed, so runs are not exactly reproducible yet.
- Checkpoints are small enough for this study repository, but larger future models should use release assets, Git LFS, or external artifact storage.
- The tests are still primarily executable study scripts rather than a fully isolated automated test suite.
