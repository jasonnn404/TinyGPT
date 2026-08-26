def get_stats(tokens):
    counts = {}

    for pair in zip(tokens, tokens[1:]):
        counts[pair] = counts.get(pair, 0) + 1

    return counts


def merge(tokens, pair, new_id):
    new_tokens = []
    i = 0

    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
            new_tokens.append(new_id)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1

    return new_tokens


class BPETokenizer:
    def __init__(self, vocab_size=500):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}


    def train(self, text):
        tokens = list(text.encode("utf-8"))

        num_merges = self.vocab_size - 256

        for i in range(num_merges):
            stats = get_stats(tokens)

            if not stats:
                break

            best_pair = max(stats, key=stats.get)
            new_id = 256 + i

            self.merges[best_pair] = new_id
            
            a, b = best_pair
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]

            tokens = merge(tokens, best_pair, new_id)


    def encode(self, text):
        tokens = list(text.encode("utf-8"))

        while len(tokens) >= 2:
            stats = get_stats(tokens)

            pair = min(
                stats,
                key=lambda p: self.merges.get(p, float("inf"))
            )

            if pair not in self.merges:
                break

            new_id = self.merges[pair]
            tokens = merge(tokens, pair, new_id)

        return tokens

    def decode(self, tokens):
        text_bytes = b"".join(self.vocab[token] for token in tokens)
        return text_bytes.decode("utf-8", errors="replace")


    def load_merges(self, merges):
        self.merges = merges
        self.vocab = {i: bytes([i]) for i in range(256)}

        for pair, new_id in self.merges.items():
            a, b = pair
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
