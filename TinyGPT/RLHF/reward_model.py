import torch
import torch.nn as nn


class RewardModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.reward_head = nn.Linear(backbone.embedding_dimension, 1)

    def forward(self, input_ids, attention_mask):
        # [B, T] -> [B, T, D]
        hidden, _ = self.backbone.forward_features(input_ids)

        # Find the last real token of every sequence
        lengths = attention_mask.sum(dim=1)
        last_indices = lengths - 1

        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        last_hidden = hidden[batch_indices, last_indices]  # [B, D]

        reward = self.reward_head(last_hidden)  # [B, 1]
        return reward.squeeze(-1)               # [B]
