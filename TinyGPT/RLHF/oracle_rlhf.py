import re

import torch
import torch.nn.functional as F

from model import TinyGPT
from bpe_tokenizer import BPETokenizer
from RLHF.critic import Critic


# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

sft_checkpoint = torch.load("checkpoints/sft_model.pt", map_location=device)
config = sft_checkpoint["config"]

tokenizer = BPETokenizer(vocab_size=config["bpe_vocab_size"])
tokenizer.load_merges(sft_checkpoint["tokenizer_merges"])


# ------------------------------------------------------------
# Build model from SFT checkpoint
# ------------------------------------------------------------

def build_sft_model():
    model = TinyGPT(
        vocab_size=config["vocab_size"],
        max_sequence_length=config["max_sequence_length"],
        embedding_dimension=config["embedding_dimension"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"]
    ).to(device)

    model.load_state_dict(sft_checkpoint["model_state_dict"])
    return model


# ------------------------------------------------------------
# Policy
# ------------------------------------------------------------

policy = build_sft_model()
policy.eval()

policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)


# ------------------------------------------------------------
# Critic
# ------------------------------------------------------------

critic_backbone = build_sft_model()
critic = Critic(critic_backbone).to(device)
critic.eval()

critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-5)


# ------------------------------------------------------------
# Reference Model
# ------------------------------------------------------------

reference = build_sft_model()
reference.eval()

for param in reference.parameters():
    param.requires_grad = False


# ------------------------------------------------------------
# Oracle Reward
# ------------------------------------------------------------

def oracle_reward(response, a, b):
    expected = a + b

    pattern = rf"\s*{a}\s*\+\s*{b}\s+is\s+{expected}\s*\.\s*"

    if re.fullmatch(pattern, response):
        return 1.0

    return -1.0


# ------------------------------------------------------------
# GAE
# ------------------------------------------------------------

def compute_gae(rewards, values, gamma=1.0, lam=0.95):
    advantages = torch.zeros_like(rewards)

    last_advantage = 0.0
    next_value = 0.0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        last_advantage = delta + gamma * lam * last_advantage

        advantages[t] = last_advantage
        next_value = values[t]

    returns = advantages + values

    return advantages, returns


# ------------------------------------------------------------
# Rollout one prompt
# ------------------------------------------------------------

def rollout_one_prompt(a, b, max_new_tokens=30, kl_coef=0.1, temperature=1.3):
    prompt = f"User: What is {a} + {b}?\nAssistant:"
    prompt_ids = tokenizer.encode(prompt)

    max_new_tokens = min(
        max_new_tokens,
        config["max_sequence_length"] - len(prompt_ids)
    )

    if max_new_tokens <= 0:
        raise ValueError("Prompt is too long for max_sequence_length.")

    context = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device
    )

    generated_tokens = []
    old_logprobs = []
    reference_logprobs = []

    # --------------------------------------------------------
    # Generate response
    # --------------------------------------------------------

    with torch.no_grad():
        for _ in range(max_new_tokens):
            policy_logits = policy(context)
            next_token_logits = policy_logits[:, -1, :]

            sampling_log_probs = F.log_softmax(
                next_token_logits / temperature,
                dim=-1
            )

            sampling_probs = torch.softmax(
                next_token_logits / temperature,
                dim=-1
            )

            next_token = torch.multinomial(
                sampling_probs,
                num_samples=1
            )

            sampled_logprob = sampling_log_probs.gather(
                -1,
                next_token
            )

            # Reference probability for SAME sampled token
            reference_logits = reference(context)

            reference_log_probs = F.log_softmax(
                reference_logits[:, -1, :] / temperature,
                dim=-1
            )

            reference_logprob = reference_log_probs.gather(
                -1,
                next_token
            )

            generated_tokens.append(next_token.item())
            old_logprobs.append(sampled_logprob.item())
            reference_logprobs.append(reference_logprob.item())

            context = torch.cat(
                [context, next_token],
                dim=1
            )

            if next_token.item() == tokenizer.eos_id:
                break

    prompt_length = len(prompt_ids)
    response_length = len(generated_tokens)

    if response_length == 0:
        raise RuntimeError("Policy generated no response tokens.")

    old_logprobs = torch.tensor(
        old_logprobs,
        dtype=torch.float32,
        device=device
    )

    reference_logprobs = torch.tensor(
        reference_logprobs,
        dtype=torch.float32,
        device=device
    )

    # --------------------------------------------------------
    # KL reward
    # --------------------------------------------------------

    kl = old_logprobs - reference_logprobs
    kl_rewards = -kl_coef * kl

    # --------------------------------------------------------
    # Decode response
    # --------------------------------------------------------

    response = tokenizer.decode(generated_tokens)

    # --------------------------------------------------------
    # Perfect oracle reward
    # --------------------------------------------------------

    final_reward = oracle_reward(response, a, b)

    # --------------------------------------------------------
    # Critic values
    # --------------------------------------------------------

    full_input = context

    with torch.no_grad():
        all_values = critic(full_input)

    response_values = all_values[
        0,
        prompt_length - 1 : prompt_length - 1 + response_length
    ]

    # --------------------------------------------------------
    # Per-token rewards
    # --------------------------------------------------------

    rewards = kl_rewards.clone()

    # Oracle reward only appears at final token
    rewards[-1] += final_reward

    # --------------------------------------------------------
    # GAE
    # --------------------------------------------------------

    advantages, returns = compute_gae(
        rewards,
        response_values
    )

    return {
        "prompt": prompt,
        "response": response,
        "full_input": full_input.squeeze(0).detach(),
        "prompt_length": prompt_length,
        "response_length": response_length,
        "old_logprobs": old_logprobs.detach(),
        "advantages": advantages.detach(),
        "returns": returns.detach(),
        "reward": final_reward,
        "kl": kl.detach()
    }


# ------------------------------------------------------------
# Build padded batch
# ------------------------------------------------------------

def build_batch(trajectories):
    batch_size = len(trajectories)

    max_total_length = max(
        len(t["full_input"])
        for t in trajectories
    )

    max_response_length = max(
        t["response_length"]
        for t in trajectories
    )

    full_inputs = torch.full(
        (batch_size, max_total_length),
        tokenizer.pad_id,
        dtype=torch.long,
        device=device
    )

    old_logprobs = torch.zeros(
        (batch_size, max_response_length),
        dtype=torch.float32,
        device=device
    )

    advantages = torch.zeros_like(old_logprobs)
    returns = torch.zeros_like(old_logprobs)
    response_mask = torch.zeros_like(old_logprobs)

    prompt_lengths = torch.zeros(
        batch_size,
        dtype=torch.long,
        device=device
    )

    response_lengths = torch.zeros(
        batch_size,
        dtype=torch.long,
        device=device
    )

    for i, trajectory in enumerate(trajectories):
        total_length = len(trajectory["full_input"])
        response_length = trajectory["response_length"]

        full_inputs[i, :total_length] = trajectory["full_input"]

        old_logprobs[i, :response_length] = trajectory["old_logprobs"]
        advantages[i, :response_length] = trajectory["advantages"]
        returns[i, :response_length] = trajectory["returns"]

        response_mask[i, :response_length] = 1.0

        prompt_lengths[i] = trajectory["prompt_length"]
        response_lengths[i] = response_length

    return {
        "full_inputs": full_inputs,
        "old_logprobs": old_logprobs,
        "advantages": advantages,
        "returns": returns,
        "response_mask": response_mask,
        "prompt_lengths": prompt_lengths,
        "response_lengths": response_lengths
    }


# ------------------------------------------------------------
# Get policy logprobs for response actions
# ------------------------------------------------------------

def get_response_logprobs(
    model,
    full_inputs,
    prompt_lengths,
    response_lengths,
    temperature=1.3
):
    logits = model(full_inputs)

    batch_size = full_inputs.size(0)
    max_response_length = int(
        response_lengths.max().item()
    )

    offsets = torch.arange(
        max_response_length,
        device=device
    ).unsqueeze(0)

    offsets = offsets.expand(
        batch_size,
        -1
    )

    valid_mask = offsets < response_lengths.unsqueeze(1)

    # Actual response token positions
    action_positions = (
        prompt_lengths.unsqueeze(1)
        + offsets
    )

    # Previous position predicts each action
    logit_positions = action_positions - 1

    safe_action_positions = torch.where(
        valid_mask,
        action_positions,
        torch.zeros_like(action_positions)
    )

    safe_logit_positions = torch.where(
        valid_mask,
        logit_positions,
        torch.zeros_like(logit_positions)
    )

    batch_indices = torch.arange(
        batch_size,
        device=device
    ).unsqueeze(1)

    batch_indices = batch_indices.expand(
        -1,
        max_response_length
    )

    action_logits = logits[
        batch_indices,
        safe_logit_positions
    ]

    actions = full_inputs[
        batch_indices,
        safe_action_positions
    ]

    log_probs = F.log_softmax(
        action_logits / temperature,
        dim=-1
    )

    action_logprobs = log_probs.gather(
        -1,
        actions.unsqueeze(-1)
    ).squeeze(-1)

    return torch.where(
        valid_mask,
        action_logprobs,
        torch.zeros_like(action_logprobs)
    )


# ------------------------------------------------------------
# Get critic values for response states
# ------------------------------------------------------------

def get_response_values(
    model,
    full_inputs,
    prompt_lengths,
    response_lengths
):
    all_values = model(full_inputs)

    batch_size = full_inputs.size(0)
    max_response_length = int(
        response_lengths.max().item()
    )

    offsets = torch.arange(
        max_response_length,
        device=device
    ).unsqueeze(0)

    offsets = offsets.expand(
        batch_size,
        -1
    )

    valid_mask = offsets < response_lengths.unsqueeze(1)

    value_positions = (
        prompt_lengths.unsqueeze(1)
        - 1
        + offsets
    )

    safe_value_positions = torch.where(
        valid_mask,
        value_positions,
        torch.zeros_like(value_positions)
    )

    batch_indices = torch.arange(
        batch_size,
        device=device
    ).unsqueeze(1)

    batch_indices = batch_indices.expand(
        -1,
        max_response_length
    )

    response_values = all_values[
        batch_indices,
        safe_value_positions
    ]

    return torch.where(
        valid_mask,
        response_values,
        torch.zeros_like(response_values)
    )


# ------------------------------------------------------------
# PPO Policy Update
# ------------------------------------------------------------

def update_policy(batch, ppo_epochs=4, clip_eps=0.2):
    full_inputs = batch["full_inputs"]

    old_logprobs = batch["old_logprobs"].detach()
    advantages = batch["advantages"].detach()

    response_mask = batch["response_mask"]

    prompt_lengths = batch["prompt_lengths"]
    response_lengths = batch["response_lengths"]

    valid = response_mask.bool()

    # --------------------------------------------------------
    # Normalize advantages over real response tokens only
    # --------------------------------------------------------

    valid_advantages = advantages[valid]

    advantage_mean = valid_advantages.mean()
    advantage_std = valid_advantages.std(unbiased=False)

    advantages = torch.where(
        valid,
        (advantages - advantage_mean) / (advantage_std + 1e-8),
        torch.zeros_like(advantages)
    )

    # --------------------------------------------------------
    # PPO epochs
    # --------------------------------------------------------

    for _ in range(ppo_epochs):
        new_logprobs = get_response_logprobs(
            policy,
            full_inputs,
            prompt_lengths,
            response_lengths
        )

        ratio = torch.exp(
            new_logprobs - old_logprobs
        )

        unclipped_objective = (
            ratio * advantages
        )

        clipped_ratio = torch.clamp(
            ratio,
            1 - clip_eps,
            1 + clip_eps
        )

        clipped_objective = (
            clipped_ratio * advantages
        )

        token_objective = torch.min(
            unclipped_objective,
            clipped_objective
        )

        policy_loss = -(
            token_objective * response_mask
        ).sum() / response_mask.sum()

        policy_optimizer.zero_grad()

        policy_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            1.0
        )

        policy_optimizer.step()

    ratio_mean = ratio[valid].mean()

    return (
        policy_loss.item(),
        ratio_mean.item()
    )


# ------------------------------------------------------------
# Critic Update
# ------------------------------------------------------------

def update_critic(batch, critic_epochs=4):
    full_inputs = batch["full_inputs"]

    returns = batch["returns"].detach()
    response_mask = batch["response_mask"]

    prompt_lengths = batch["prompt_lengths"]
    response_lengths = batch["response_lengths"]

    for _ in range(critic_epochs):
        new_values = get_response_values(
            critic,
            full_inputs,
            prompt_lengths,
            response_lengths
        )

        value_errors = (
            new_values - returns
        ) ** 2

        value_loss = (
            value_errors * response_mask
        ).sum() / response_mask.sum()

        critic_optimizer.zero_grad()

        value_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            critic.parameters(),
            1.0
        )

        critic_optimizer.step()

    return value_loss.item()


# ------------------------------------------------------------
# Only the 5 held-out SFT mistakes
# ------------------------------------------------------------

prompt_pool = [
    (1, 1),
    (2, 9),
    (4, 7),
    (6, 6),
    (7, 8)
]


# ------------------------------------------------------------
# Oracle PPO Training Loop
# ------------------------------------------------------------

num_iterations = 100

for iteration in range(num_iterations):
    # --------------------------------------------------------
    # 1. Fresh rollout for all 5 questions
    # --------------------------------------------------------

    trajectories = []

    for a, b in prompt_pool:
        trajectory = rollout_one_prompt(
            a,
            b
        )

        trajectories.append(
            trajectory
        )

    # --------------------------------------------------------
    # 2. Build padded batch
    # --------------------------------------------------------

    batch = build_batch(
        trajectories
    )

    # --------------------------------------------------------
    # 3. PPO Policy update
    # --------------------------------------------------------

    policy_loss, ratio_mean = update_policy(
        batch
    )

    # --------------------------------------------------------
    # 4. Critic update
    # --------------------------------------------------------

    value_loss = update_critic(
        batch
    )

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    correct_rollouts = sum(
        trajectory["reward"] > 0
        for trajectory in trajectories
    )

    mean_reward = sum(
        trajectory["reward"]
        for trajectory in trajectories
    ) / len(trajectories)

    mean_kl = sum(
        trajectory["kl"].mean().item()
        for trajectory in trajectories
    ) / len(trajectories)

    print(
        f"Iteration {iteration + 1:3d} | "
        f"oracle correct {correct_rollouts}/5 | "
        f"reward {mean_reward:+.2f} | "
        f"policy loss {policy_loss:.4f} | "
        f"value loss {value_loss:.4f} | "
        f"ratio {ratio_mean:.4f} | "
        f"KL {mean_kl:.4f}"
    )

    # --------------------------------------------------------
    # Print all 5 responses every 10 iterations
    # --------------------------------------------------------

    if (iteration + 1) % 10 == 0:
        print()

        for trajectory in trajectories:
            print(trajectory["prompt"])
            print(trajectory["response"])
            print(f"reward: {trajectory['reward']:+.0f}")
            print()

    # --------------------------------------------------------
    # Save checkpoint every 10 iterations
    # --------------------------------------------------------

    if (iteration + 1) % 10 == 0:
        torch.save(
            {
                "policy_state_dict": policy.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "config": config,
                "tokenizer_merges": sft_checkpoint["tokenizer_merges"],
                "iteration": iteration + 1
            },
            "checkpoints/oracle_rlhf_model.pt"
        )

        print("Saved oracle RLHF checkpoint.")


print("\nOracle RLHF training complete.")