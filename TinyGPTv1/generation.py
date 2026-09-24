import torch
import torch.nn.functional as F


def generate(
    model,
    tokenizer,
    prompt,
    block_size,
    max_new_tokens=100,
    temperature=0.8,
    k=5
):

    device = next(model.parameters()).device

    idx = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=device
    )

    model.eval()

    with torch.no_grad():

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -block_size:]

            logits = model(idx_cond)

            next_token_logits = logits[:, -1, :]

            topk_values, topk_indices = torch.topk(
                next_token_logits,
                k
            )

            topk_probs = F.softmax(
                topk_values / temperature,
                dim=-1
            )

            choice = torch.multinomial(
                topk_probs,
                num_samples=1
            )

            next_token = torch.gather(
                topk_indices,
                dim=-1,
                index=choice
            )

            idx = torch.cat(
                [idx, next_token],
                dim=1
            )

    return tokenizer.decode(
        idx[0].tolist()
    )
