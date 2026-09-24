# Study Log

This is a running record of what I built, what I learned, and what I want to test next. Entries should capture reasoning and results, not just code changes.

## 2026-08-25 — TinyGPT v1: character-level model

### Built

- Character tokenizer with encode/decode mappings learned from the corpus.
- Next-token batches with a 90/10 train/validation split.
- Scaled dot-product causal self-attention and multi-head attention.
- Pre-normalized transformer blocks with residual connections and a feed-forward network.
- Token and position embeddings, training loop, validation, checkpoints, and top-k generation.

### Key ideas

- The causal mask prevents each position from reading future tokens.
- Splitting the embedding dimension across heads lets the model learn different attention patterns.
- The language-model objective predicts the next token at every position in the context window.
- Validation loss is needed to distinguish learning from memorizing the training split.

### Snapshot configuration

- Tokenizer: characters from the training corpus
- Context length: 64
- Embedding dimension: 64
- Attention heads: 4
- Transformer layers: 2
- Batch size: 4
- Optimizer: AdamW
- Learning rate: 0.001

## 2026-08-26 — TinyGPT: byte-pair encoding

### Built

- A byte-level BPE tokenizer starting with 256 byte tokens.
- Repeated merging of the most frequent adjacent token pair.
- Checkpoint persistence for both the model and learned BPE merges.
- A comparison script for character-token and BPE-token sequence lengths.

### Key ideas

- Character tokenization is simple but creates long sequences.
- BPE learns reusable multi-byte pieces and can represent any UTF-8 input from its byte base vocabulary.
- Tokenizer state is part of the model artifact: the same merges must be restored for generation.
- Tokenization changes vocabulary size and sequence length, affecting both compute and what the model can learn.

### Snapshot configuration

- BPE vocabulary size: 500
- Context length: 64
- Embedding dimension: 64
- Attention heads: 4
- Transformer layers: 2
- Batch size: 4
- Optimizer: AdamW
- Learning rate: 0.001

## Next experiments

- Fix a random seed and compare repeated runs.
- Compare validation loss and wall-clock time for character and BPE tokenization.
- Sweep context length, embedding dimension, head count, and layer count one variable at a time.
- Add dropout and compare overfitting behavior.
- Add command-line configuration so experiments do not require editing source files.
- Track parameter count, tokens processed per second, and generated samples.
