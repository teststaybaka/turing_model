"""Parallel dense actor-critic pretraining on arithmetic tick trajectories.

The tape states and previous actions remain teacher-forced, so this is a
contextual-bandit form of dense RL rather than an on-policy environment
rollout. At every valid tick, each sampled move/write factor receives +1 for
matching its supervised target and -1 otherwise. Random latent inputs make the
whole chunk independent of explicit latent feedback.
"""

import math
import os
import time

import torch
import torch.nn.functional as F

from tick_arith_data_loader import (
    ArithmeticTickDataLoader,
    ArithmeticTickDataset,
    MOVE_VOCAB_SIZE,
    READ_VOCAB_SIZE,
    TEST_LONG_ADD,
    TEST_LONG_MUL,
    TRAIN_ADD,
    TRAIN_MUL,
    VAL_ADD,
    VAL_MUL,
    WRITE_VOCAB_SIZE,
)
from tick_rl_mamba3_model import GPT, GPTConfig, NUM_ACTION_FACTORS


MICRO_BATCH_SIZE = 128
GRAD_ACCUM_STEPS = 1
EVAL_INTERVAL = 50
DEVICE = "cuda"
COMPILE = True
TASKS = ["add", "mul"]

REWARD_CORRECT = 1.0
REWARD_INCORRECT = -1.0
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01

max_lr = 3e-4
min_lr = max_lr * 0.1
warmup_steps = 30

config = GPTConfig(
    block_size=32,
    read_vocab_size=READ_VOCAB_SIZE,
    move_vocab_size=MOVE_VOCAB_SIZE,
    write_vocab_size=WRITE_VOCAB_SIZE,
    token_embd=32,
    n_layers=4,
    n_heads=4,
    n_embd=64,
)

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "tick_arith_dense_rl_mamba3_log.txt")


def get_lr(step, total_steps):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def configure_optimizer(model, weight_decay):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for p in param_dict.values() if p.ndim >= 2]
    no_decay_params = [p for p in param_dict.values() if p.ndim < 2]
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=max_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def unpack_batch(batch, device):
    (
        head0_reads,
        head1_reads,
        prev_head0_moves,
        prev_head1_moves,
        prev_head0_writes,
        prev_head1_writes,
        target_head0_moves,
        target_head1_moves,
        target_head0_writes,
        target_head1_writes,
        loss_mask,
    ) = [x.to(device) for x in batch]

    reads = torch.stack((head0_reads, head1_reads), dim=-1)
    prev_actions = torch.stack(
        (
            prev_head0_moves,
            prev_head1_moves,
            prev_head0_writes,
            prev_head1_writes,
        ),
        dim=-1,
    )
    targets = (
        target_head0_moves,
        target_head1_moves,
        target_head0_writes,
        target_head1_writes,
    )
    return reads, prev_actions, targets, loss_mask


def masked_mean(values, mask):
    mask = mask.to(dtype=values.dtype)
    denominator = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denominator


def sample_categorical(logits):
    """Sample every B x T policy position without a Python token loop."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    sampled = torch.multinomial(
        probs.reshape(-1, probs.size(-1)), num_samples=1
    ).view(logits.shape[:-1])
    sampled_log_probs = log_probs.gather(
        dim=-1, index=sampled.unsqueeze(-1)
    ).squeeze(-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return sampled, sampled_log_probs, entropy


def dense_actor_critic_loss(logits, values, targets, loss_mask):
    """One-step factorized actor-critic loss on fixed teacher states."""
    assert len(logits) == len(targets) == NUM_ACTION_FACTORS
    assert values.shape == (*loss_mask.shape, NUM_ACTION_FACTORS)

    sampled_actions = []
    sampled_log_probs = []
    entropies = []
    rewards = []

    for factor_logits, target in zip(logits, targets):
        sampled, log_prob, entropy = sample_categorical(factor_logits)
        reward = torch.where(
            sampled.eq(target),
            sampled.new_tensor(REWARD_CORRECT, dtype=torch.float32),
            sampled.new_tensor(REWARD_INCORRECT, dtype=torch.float32),
        )
        sampled_actions.append(sampled)
        sampled_log_probs.append(log_prob)
        entropies.append(entropy)
        rewards.append(reward)

    actions = torch.stack(sampled_actions, dim=-1)
    log_probs = torch.stack(sampled_log_probs, dim=-1)
    entropies = torch.stack(entropies, dim=-1)
    rewards = torch.stack(rewards, dim=-1).to(dtype=values.dtype)
    factor_mask = loss_mask.unsqueeze(-1).expand_as(values)

    # The critic is a baseline only; actor gradients must not pass through it.
    advantages = (rewards - values).detach()
    policy_loss = masked_mean(-log_probs * advantages, factor_mask)
    value_loss = masked_mean((values - rewards).square(), factor_mask)
    entropy = masked_mean(entropies, factor_mask)
    loss = policy_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF * entropy

    target_actions = torch.stack(targets, dim=-1)
    factor_correct = actions.eq(target_actions)
    tick_correct = factor_correct.all(dim=-1)
    factor_accuracy = masked_mean(factor_correct.float(), factor_mask)
    tick_accuracy = masked_mean(tick_correct.float(), loss_mask)
    mean_reward = masked_mean(rewards, factor_mask)

    metrics = {
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": entropy.detach(),
        "reward": mean_reward.detach(),
        "factor_acc": factor_accuracy.detach(),
        "tick_acc": tick_accuracy.detach(),
    }
    return loss, metrics


def forward_chunked(model, reads, prev_actions, targets, loss_mask, chunk_size):
    """Run sequence chunks with Mamba state carry and parallel RL sampling."""
    B, T = loss_mask.size()
    states = None
    total_loss = torch.tensor(0.0, device=loss_mask.device)
    totals = None
    total_masked = 0

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        latent = torch.randn(
            B,
            end - start,
            config.n_embd,
            device=loss_mask.device,
        )
        logits, values, _, states = model(
            reads[:, start:end].contiguous(),
            prev_actions[:, start:end].contiguous(),
            latent=latent,
            states=states,
        )
        chunk_mask = loss_mask[:, start:end].contiguous()
        n_masked = int(chunk_mask.sum().item())
        if n_masked == 0:
            continue

        chunk_targets = tuple(
            target[:, start:end].contiguous() for target in targets
        )
        chunk_loss, metrics = dense_actor_critic_loss(
            logits, values, chunk_targets, chunk_mask
        )
        total_loss = total_loss + chunk_loss * n_masked
        if totals is None:
            totals = {name: value * n_masked for name, value in metrics.items()}
        else:
            for name, value in metrics.items():
                totals[name] = totals[name] + value * n_masked
        total_masked += n_masked

    if total_masked == 0:
        return total_loss, {
            name: total_loss.detach()
            for name in (
                "policy_loss",
                "value_loss",
                "entropy",
                "reward",
                "factor_acc",
                "tick_acc",
            )
        }
    metrics = {name: value / total_masked for name, value in totals.items()}
    return total_loss / total_masked, metrics


def factorized_nll(logits, targets, mask):
    losses = []
    for factor_logits, target in zip(logits, targets):
        per_tick = F.cross_entropy(
            factor_logits.reshape(-1, factor_logits.size(-1)),
            target.reshape(-1),
            reduction="none",
        ).view_as(target)
        losses.append(masked_mean(per_tick, mask))
    return sum(losses) / len(losses)


def evaluate(model, loader):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    factor_correct = 0.0
    tick_correct = 0.0

    with torch.no_grad():
        for batch in loader:
            reads, prev_actions, targets, loss_mask = unpack_batch(batch, DEVICE)
            B, T = loss_mask.size()
            states = None
            logit_parts = [[] for _ in targets]

            for start in range(0, T, config.block_size):
                end = min(start + config.block_size, T)
                latent = torch.randn(
                    B,
                    end - start,
                    config.n_embd,
                    device=loss_mask.device,
                )
                logits, _, _, states = model(
                    reads[:, start:end].contiguous(),
                    prev_actions[:, start:end].contiguous(),
                    latent=latent,
                    states=states,
                )
                for factor, factor_logits in enumerate(logits):
                    logit_parts[factor].append(factor_logits)

            all_logits = tuple(
                torch.cat(parts, dim=1) for parts in logit_parts
            )
            n_tokens = int(loss_mask.sum().item())
            nll = factorized_nll(all_logits, targets, loss_mask)
            total_nll += nll.item() * n_tokens
            total_tokens += n_tokens

            all_factor_correct = []
            for factor_logits, target in zip(all_logits, targets):
                all_factor_correct.append(
                    factor_logits.argmax(dim=-1).eq(target)
                )
            all_factor_correct = torch.stack(all_factor_correct, dim=-1)
            factor_mask = loss_mask.unsqueeze(-1).expand_as(all_factor_correct)
            factor_correct += (
                all_factor_correct.float() * factor_mask
            ).sum().item()
            tick_correct += (
                all_factor_correct.all(dim=-1).float() * loss_mask
            ).sum().item()

    return {
        "nll": total_nll / total_tokens,
        "factor_acc": factor_correct / (total_tokens * NUM_ACTION_FACTORS),
        "tick_acc": tick_correct / total_tokens,
    }


def train():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    train_dataset = ArithmeticTickDataset(
        add_items=TRAIN_ADD,
        mul_items=TRAIN_MUL,
        tasks=TASKS,
        shuffle_seed=42,
    )
    val_dataset = ArithmeticTickDataset(
        add_items=VAL_ADD,
        mul_items=VAL_MUL,
        tasks=TASKS,
    )
    test_dataset = ArithmeticTickDataset(
        add_items=TEST_LONG_ADD,
        mul_items=TEST_LONG_MUL,
        tasks=TASKS,
    )

    train_loader = ArithmeticTickDataLoader(
        train_dataset,
        batch_size=MICRO_BATCH_SIZE,
        pad_to_multiple=config.block_size,
    )
    val_loader = ArithmeticTickDataLoader(
        val_dataset,
        batch_size=MICRO_BATCH_SIZE,
        pad_to_multiple=config.block_size,
    )
    test_loader = ArithmeticTickDataLoader(
        test_dataset,
        batch_size=MICRO_BATCH_SIZE,
        pad_to_multiple=config.block_size,
    )
    total_steps = len(train_loader) // GRAD_ACCUM_STEPS

    model = GPT(config).to(DEVICE)
    raw_model = model
    if COMPILE:
        model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    print(f"Config: {config}")
    print(f"Tasks: {TASKS}")
    print(f"Compile: {COMPILE}")
    print("Objective: parallel dense factorized actor-critic")
    print(
        f"Reward: correct {REWARD_CORRECT:+.1f}, "
        f"incorrect {REWARD_INCORRECT:+.1f}"
    )
    print(
        f"Value coefficient: {VALUE_LOSS_COEF} | "
        f"entropy coefficient: {ENTROPY_COEF}"
    )
    print(f"Device: {DEVICE}")
    print(f"Train: {len(train_dataset)} examples")
    print(f"Total optimizer steps: {total_steps}")
    print()

    optimizer = configure_optimizer(model, weight_decay=0.1)
    with open(log_file, "w"):
        pass

    batch_iter = iter(train_loader)
    for step in range(total_steps):
        if step % EVAL_INTERVAL == 0:
            val = evaluate(raw_model, val_loader)
            print(
                f"step {step:4d}/{total_steps} | val_nll {val['nll']:.4f} | "
                f"factor_acc {val['factor_acc']:.4f} | "
                f"tick_acc {val['tick_acc']:.4f}"
            )
            with open(log_file, "a") as f:
                f.write(
                    f"{step} val nll {val['nll']:.4f} "
                    f"factor_acc {val['factor_acc']:.4f} "
                    f"tick_acc {val['tick_acc']:.4f}\n"
                )

        lr = get_lr(step, total_steps)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = lr

        model.train()
        optimizer.zero_grad()
        accumulated = None
        t0 = time.time()

        for _ in range(GRAD_ACCUM_STEPS):
            batch = next(batch_iter)
            reads, prev_actions, targets, loss_mask = unpack_batch(batch, DEVICE)
            loss, metrics = forward_chunked(
                model,
                reads,
                prev_actions,
                targets,
                loss_mask,
                config.block_size,
            )
            (loss / GRAD_ACCUM_STEPS).backward()
            if accumulated is None:
                accumulated = {
                    "loss": loss.detach() / GRAD_ACCUM_STEPS,
                    **{
                        name: value / GRAD_ACCUM_STEPS
                        for name, value in metrics.items()
                    },
                }
            else:
                accumulated["loss"] += loss.detach() / GRAD_ACCUM_STEPS
                for name, value in metrics.items():
                    accumulated[name] += value / GRAD_ACCUM_STEPS

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        dt = time.time() - t0
        print(
            f"step {step + 1:4d}/{total_steps} | "
            f"loss {accumulated['loss']:.4f} | "
            f"policy {accumulated['policy_loss']:.4f} | "
            f"value {accumulated['value_loss']:.4f} | "
            f"reward {accumulated['reward']:+.4f} | "
            f"factor_acc {accumulated['factor_acc']:.4f} | "
            f"tick_acc {accumulated['tick_acc']:.4f} | "
            f"entropy {accumulated['entropy']:.4f} | "
            f"norm {norm:.4f} | lr {lr:.2e} | dt {dt:.2f}s"
        )

    val = evaluate(raw_model, val_loader)
    test = evaluate(raw_model, test_loader)
    print(
        f"\nVal:         nll {val['nll']:.4f} | "
        f"factor_acc {val['factor_acc']:.4f} | "
        f"tick_acc {val['tick_acc']:.4f}"
    )
    print(
        f"Test (long): nll {test['nll']:.4f} | "
        f"factor_acc {test['factor_acc']:.4f} | "
        f"tick_acc {test['tick_acc']:.4f}"
    )
    with open(log_file, "a") as f:
        f.write(
            f"final val nll {val['nll']:.4f} "
            f"factor_acc {val['factor_acc']:.4f} "
            f"tick_acc {val['tick_acc']:.4f}\n"
        )
        f.write(
            f"final test nll {test['nll']:.4f} "
            f"factor_acc {test['factor_acc']:.4f} "
            f"tick_acc {test['tick_acc']:.4f}\n"
        )

    checkpoint_path = "log/tick_arith_dense_rl_mamba3_model.pt"
    torch.save({"model": raw_model.state_dict(), "config": config}, checkpoint_path)
    print(f"Saved model to {checkpoint_path}")


if __name__ == "__main__":
    train()
