import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.value_head = nn.Linear(backbone.embedding_dimension, 1)

    def forward(self, input_ids):
        hidden, _ = self.backbone.forward_features(input_ids)  # [B, T, D]
        values = self.value_head(hidden)                    # [B, T, 1]
        return values.squeeze(-1)                           # [B, T]
