import torch
import math

torch.manual_seed(42)

batch_size = 1
sequence_length = 4
embedding_dimension = 8

x = torch.randn(
    batch_size,
    sequence_length,
    embedding_dimension,
)

W_q = torch.randn(embedding_dimension, embedding_dimension)
W_k = torch.randn(embedding_dimension, embedding_dimension)
W_v = torch.randn(embedding_dimension, embedding_dimension)

Q = x @ W_q
K = x @ W_k
V = x @ W_v

# print("Q:", Q.shape)

scores = Q @ K.transpose(-2, -1) / math.sqrt(embedding_dimension)

# print("scores:", scores.shape)

mask = torch.tril(
    torch.ones(sequence_length, sequence_length)
)

scores = scores.masked_fill(
    mask == 0,
    float("-inf")
)

attention_weights = torch.softmax(scores, dim=-1)

# print("attention weights:", attention_weights.shape)

output = attention_weights @ V

# print("output:", output.shape)
# print("causal mask:")
# print(mask)

# print("\nmasked scores:")
# print(scores)

# print("\nattention weights:")
# print(attention_weights)



# Multi-head attention

num_heads = 2
head_dimension = embedding_dimension // num_heads

Q = x @ W_q
K = x @ W_k
V = x @ W_v

Q = Q.view(
    batch_size,
    sequence_length,
    num_heads,
    head_dimension
).transpose(1, 2)

K = K.view(
    batch_size,
    sequence_length,
    num_heads,
    head_dimension
).transpose(1, 2)

V = V.view(
    batch_size,
    sequence_length,
    num_heads,
    head_dimension
).transpose(1, 2)

scores = Q @ K.transpose(-2, -1) / math.sqrt(head_dimension)

mask = torch.tril(
    torch.ones(sequence_length, sequence_length)
)

scores = scores.masked_fill(
    mask == 0,
    float("-inf")
)

attention_weights = torch.softmax(scores, dim=-1)

output = attention_weights @ V

# print("Q:", Q.shape)
# print("scores:", scores.shape)
# print("attention weights:", attention_weights.shape)
# print("output:", output.shape)

output = output.transpose(1, 2)

# print("output after transpose:", output.shape)

output = output.reshape(
    batch_size,
    sequence_length,
    embedding_dimension
)

# print("concatenated output:", output.shape)

W_o = torch.randn(
    embedding_dimension,
    embedding_dimension
)

output = output @ W_o

print(output.shape)