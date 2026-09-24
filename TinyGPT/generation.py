import torch
from .kv_cache import KVCache


@torch.no_grad()
def generate_greedy(model, input_ids, max_new_tokens):
    generated = input_ids

    for _ in range(max_new_tokens):
        logits = model(generated)

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True
        )

        generated = torch.cat(
            [generated, next_token],
            dim=1
        )

    return generated


@torch.no_grad()
def generate_greedy_cached(model, input_ids, max_new_tokens):
    logits, kv_cache = model(
        input_ids,
        use_cache=True
    )

    generated = input_ids

    for i in range(max_new_tokens):
        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True
        )

        generated = torch.cat(
            [generated, next_token],
            dim=1
        )

        if i == max_new_tokens - 1:
            break

        logits, kv_cache = model(
            next_token,
            past_kv=kv_cache,
            use_cache=True
        )

    return generated


@torch.no_grad()
def generate_greedy_preallocated(model, input_ids, max_new_tokens):
    batch_size = input_ids.shape[0]
    attention = model.blocks[0].attention

    kv_cache = KVCache(
        num_layers=len(model.blocks),
        batch_size=batch_size,
        num_heads=attention.num_heads,
        max_sequence_length=model.position_embedding.num_embeddings,
        head_dimension=attention.head_dimension,
        device=input_ids.device,
        dtype=next(model.parameters()).dtype
    )

    logits = model(input_ids, kv_cache=kv_cache)
    generated = input_ids

    for i in range(max_new_tokens):

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        if i == max_new_tokens - 1:
            break

        logits = model(next_token, kv_cache=kv_cache)

    return generated
