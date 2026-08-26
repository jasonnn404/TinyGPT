import math

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dimension, num_heads):
        super().__init__()

        assert embedding_dimension % num_heads == 0

        self.embedding_dimension = embedding_dimension
        self.num_heads = num_heads
        self.head_dimension = embedding_dimension // num_heads

        self.q_proj = nn.Linear(
            embedding_dimension,
            embedding_dimension,
            bias=False
        )

        self.k_proj = nn.Linear(
            embedding_dimension,
            embedding_dimension,
            bias=False
        )

        self.v_proj = nn.Linear(
            embedding_dimension,
            embedding_dimension,
            bias=False
        )

        self.out_proj = nn.Linear(
            embedding_dimension,
            embedding_dimension,
            bias=False
        )

    def forward(self, x):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        batch_size, sequence_length, _ = x.shape

        Q = Q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dimension
        ).transpose(1, 2)

        K = K.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dimension
        ).transpose(1, 2)

        V = V.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dimension
        ).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1)

        scores = scores / math.sqrt(self.head_dimension)

        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device
            )
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        attention_weights = torch.softmax(scores, dim=-1)

        output = attention_weights @ V

        output = output.transpose(1, 2)

        output = output.reshape(
            batch_size,
            sequence_length,
            self.embedding_dimension
        )

        output = self.out_proj(output)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dimension, num_heads):
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_dimension)

        self.attention = MultiHeadAttention(
            embedding_dimension,
            num_heads
        )

        self.norm2 = nn.LayerNorm(embedding_dimension)

        self.ffn = nn.Sequential(
            nn.Linear(
                embedding_dimension,
                4 * embedding_dimension
            ),
            nn.ReLU(),
            nn.Linear(
                4 * embedding_dimension,
                embedding_dimension
            )
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))

        x = x + self.ffn(self.norm2(x))

        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_sequence_length,
        embedding_dimension,
        num_heads,
        num_layers
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dimension
        )

        self.position_embedding = nn.Embedding(
            max_sequence_length,
            embedding_dimension
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dimension,
                num_heads
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(
            embedding_dimension
        )

        self.lm_head = nn.Linear(
            embedding_dimension,
            vocab_size,
            bias=False
        )

    def forward(self, token_ids):
        batch_size, sequence_length = token_ids.shape

        positions = torch.arange(
            sequence_length,
            device=token_ids.device
        )

        token_emb = self.token_embedding(token_ids)
        pos_emb = self.position_embedding(positions)

        x = token_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits