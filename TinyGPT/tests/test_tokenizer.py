from pathlib import Path

from TinyGPT.bpe_tokenizer import BPETokenizer
from TinyGPTv1.tokenizer import CharTokenizer

data_path = Path(__file__).parent.parent / "data.txt"

with open(data_path, "r") as f:
    text = f.read(10_000)

# Character tokenizer
char_tokenizer = CharTokenizer(text=text)
char_tokens = char_tokenizer.encode(text)


# BPE tokenizer
bpe_tokenizer = BPETokenizer(vocab_size=500)
bpe_tokenizer.train(text)
bpe_tokens = bpe_tokenizer.encode(text)


print("Character tokens:", len(char_tokens))
print("BPE tokens:", len(bpe_tokens))

print(
    "Compression ratio:",
    len(char_tokens) / len(bpe_tokens)
)
