import torch


class TextData:
    def __init__(self, text, tokenizer, batch_size, block_size):
        self.batch_size = batch_size
        self.block_size = block_size

        data = torch.tensor(
            tokenizer.encode(text),
            dtype=torch.long
        )

        n = int(0.9 * len(data))

        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split):
        source = (
            self.train_data
            if split == "train"
            else self.val_data
        )

        starts = torch.randint(
            0,
            len(source) - self.block_size,
            (self.batch_size,)
        )

        x = torch.stack([
            source[start:start + self.block_size]
            for start in starts
        ])

        y = torch.stack([
            source[start + 1:start + self.block_size + 1]
            for start in starts
        ])

        return x, y