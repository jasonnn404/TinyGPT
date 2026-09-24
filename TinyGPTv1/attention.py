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




vocab_size = 4
max_sequence_length = 4
embedding_dimension = 8
num_heads = 2
num_layers = 2


model = TinyGPT(
    vocab_size=vocab_size,
    max_sequence_length=max_sequence_length,
    embedding_dimension=embedding_dimension,
    num_heads=num_heads,
    num_layers=num_layers
)



input_ids = torch.tensor([
    [0, 1, 2]
])

targets = torch.tensor([
    [1, 2, 3]
])

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3
)

# Training loop
for step in range(500):
    optimizer.zero_grad()

    logits = model(input_ids)

    loss = loss_function(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1)
    )

    loss.backward()

    optimizer.step()

    if step % 50 == 0:
        print(
            "step:",
            step,
            "loss:",
            loss.item()
        )

id_to_word = {
    0: "I",
    1: "love",
    2: "machine",
    3: "learning"
}

generated = torch.tensor([[0]])

for _ in range(3):
    logits = model(generated)

    next_token_logits = logits[:, -1, :]

    next_token = torch.argmax(
        next_token_logits,
        dim=-1,
        keepdim=True
    )

    generated = torch.cat(
        [generated, next_token],
        dim=1
    )

words = [
    id_to_word[token.item()]
    for token in generated[0]
]

print("Generated:")
print(" ".join(words))



