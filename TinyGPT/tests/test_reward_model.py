import torch

from TinyGPT.model import TinyGPT
from TinyGPT.RLHF.reward_model import RewardModel


vocab_size = 500
max_sequence_length = 64
embedding_dimension = 128
num_heads = 4
num_layers = 4


backbone = TinyGPT(
    vocab_size=vocab_size,
    max_sequence_length=max_sequence_length,
    embedding_dimension=embedding_dimension,
    num_heads=num_heads,
    num_layers=num_layers,
)


reward_model = RewardModel(backbone)


input_ids = torch.randint(
    0,
    vocab_size,
    (4, 32)
)

attention_mask = torch.ones_like(input_ids)
rewards = reward_model(input_ids, attention_mask)


print("input shape:", input_ids.shape)
print("reward shape:", rewards.shape)
print("rewards:", rewards)
