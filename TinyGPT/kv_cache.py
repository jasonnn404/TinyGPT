import torch


class KVCache:
    def __init__(
        self,
        num_layers,
        batch_size,
        num_heads,
        max_sequence_length,
        head_dimension,
        device,
        dtype
    ):
        self.max_sequence_length = max_sequence_length
        self.current_length = 0

        self.keys = []
        self.values = []

        for _ in range(num_layers):
            self.keys.append(torch.empty(
                batch_size,
                num_heads,
                max_sequence_length,
                head_dimension,
                device=device,
                dtype=dtype
            ))

            self.values.append(torch.empty(
                batch_size,
                num_heads,
                max_sequence_length,
                head_dimension,
                device=device,
                dtype=dtype
            ))

    def update(self, layer_idx, new_k, new_v):
        num_new_tokens = new_k.shape[2]

        start = self.current_length
        end = start + num_new_tokens

        if end > self.max_sequence_length:
            raise ValueError("KV cache exceeded max sequence length")

        self.keys[layer_idx][:, :, start:end, :].copy_(new_k)
        self.values[layer_idx][:, :, start:end, :].copy_(new_v)

        k = self.keys[layer_idx][:, :, :end, :]
        v = self.values[layer_idx][:, :, :end, :]

        return k, v

    def advance(self, num_tokens):
        self.current_length += num_tokens

    def reset(self):
        self.current_length = 0