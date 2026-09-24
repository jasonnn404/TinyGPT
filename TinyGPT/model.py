import math

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dimension, num_heads, dropout=0.1):
        super().__init__()

        assert embedding_dimension % num_heads == 0

        self.embedding_dimension = embedding_dimension
        self.num_heads = num_heads
        self.head_dimension = embedding_dimension // num_heads

        self.q_proj = nn.Linear(embedding_dimension, embedding_dimension, bias=False)
        self.k_proj = nn.Linear(embedding_dimension, embedding_dimension, bias=False)
        self.v_proj = nn.Linear(embedding_dimension, embedding_dimension, bias=False)
        self.out_proj = nn.Linear(embedding_dimension, embedding_dimension, bias=False)

        self.attention_dropout = nn.Dropout(dropout)

    def forward(self, x, past_kv=None, kv_cache=None, layer_idx=None):
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

        # New: preallocated KV cache
        if kv_cache is not None:
            K, V = kv_cache.update(
                layer_idx,
                K,
                V
            )

        # Old: naive KV cache
        elif past_kv is not None:
            past_k, past_v = past_kv

            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)

        present_kv = (K, V)

        # Normal attention
        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dimension)

        # Causal mask that works for prefill + decode
        key_length = K.shape[2]
        past_length = key_length - sequence_length

        query_positions = torch.arange(
            past_length,
            past_length + sequence_length,
            device=x.device
        ).unsqueeze(1)

        key_positions = torch.arange(
            key_length,
            device=x.device
        ).unsqueeze(0)

        mask = key_positions <= query_positions

        scores = scores.masked_fill(
            ~mask,
            float("-inf")
        )

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        output = attention_weights @ V

        output = output.transpose(1, 2)
        output = output.reshape(
            batch_size,
            sequence_length,
            self.embedding_dimension
        )

        output = self.out_proj(output)

        return output, present_kv


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dimension, num_heads, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_dimension)

        self.attention = MultiHeadAttention(
            embedding_dimension,
            num_heads,
            dropout
        )

        self.norm2 = nn.LayerNorm(embedding_dimension)

        self.ffn = nn.Sequential(
            nn.Linear(
                embedding_dimension,
                4 * embedding_dimension
            ),
            nn.GELU(),
            nn.Linear(
                4 * embedding_dimension,
                embedding_dimension
            ),
            nn.Dropout(dropout)
        )

    def forward(self, x, past_kv=None, kv_cache=None, layer_idx=None):
        attention_output, present_kv = self.attention(
            self.norm1(x),
            past_kv=past_kv,
            kv_cache=kv_cache,
            layer_idx=layer_idx
        )

        x = x + attention_output
        x = x + self.ffn(self.norm2(x))

        return x, present_kv


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_sequence_length,
        embedding_dimension,
        num_heads,
        num_layers,
        dropout=0.1
    ):
        super().__init__()

        self.embedding_dimension = embedding_dimension

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
                num_heads,
                dropout
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

        self.apply(self._init_weights)

        self.lm_head.weight = self.token_embedding.weight

        self.embedding_dropout = nn.Dropout(dropout)




    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )


    def forward_features(self, token_ids, past_kv=None, kv_cache=None):
        batch_size, sequence_length = token_ids.shape

        if kv_cache is not None:
            past_length = kv_cache.current_length
        elif past_kv is not None:
            past_length = past_kv[0][0].shape[2]
        else:
            past_length = 0

        positions = torch.arange(
            past_length,
            past_length + sequence_length,
            device=token_ids.device
        )

        token_emb = self.token_embedding(token_ids)
        pos_emb = self.position_embedding(positions)

        x = token_emb + pos_emb
        x = self.embedding_dropout(x)

        new_kv = []

        for i, block in enumerate(self.blocks):
            layer_past_kv = None if past_kv is None else past_kv[i]

            x, layer_present_kv = block(
                x,
                past_kv=layer_past_kv,
                kv_cache=kv_cache,
                layer_idx=i
            )

            new_kv.append(layer_present_kv)

        # Very important:
        # only advance after every layer wrote to the same positions
        if kv_cache is not None:
            kv_cache.advance(sequence_length)

        x = self.final_norm(x)

        return x, new_kv


    def forward(self, token_ids, past_kv=None, kv_cache=None, use_cache=False):
        x, new_kv = self.forward_features(
            token_ids,
            past_kv=past_kv,
            kv_cache=kv_cache
        )

        logits = self.lm_head(x)

        if use_cache:
            return logits, new_kv

        return logits
