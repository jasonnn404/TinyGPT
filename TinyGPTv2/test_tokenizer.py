from pathlib import Path

from TinyGPT.tokenizer import CharTokenizer
from TinyGPTv2.bpe_tokenizer import BPETokenizer

data_path = Path(__file__).parent / "data.txt"

with open(data_path, "r") as f:
    text = f.read()

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
