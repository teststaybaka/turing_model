"""Supervised arithmetic tick-level Turing-machine training with Mamba-3.

There is no READ action. At each tick the model observes both tape heads, the
previous factorized actions, and the previous latent output, then predicts the
next factorized actions.
"""

import os
import time
import math
import torch
import torch.nn.functional as F

from tick_arith_data_loader import (
    ArithmeticTickDataset,
    ArithmeticTickDataLoader,
    TRAIN_ADD,
    VAL_ADD,
    TEST_LONG_ADD,
    TRAIN_MUL,
    VAL_MUL,
    TEST_LONG_MUL,
    READ_VOCAB_SIZE,
    MOVE_VOCAB_SIZE,
    WRITE_VOCAB_SIZE,
)
from tick_mamba3_model import GPT, GPTConfig


MICRO_BATCH_SIZE = 128
GRAD_ACCUM_STEPS = 1
EVAL_INTERVAL = 50
DEVICE = "cuda"
COMPILE = True
DETACH_LATENT = False
TASKS = ["add", "mul"]

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
log_file = os.path.join(log_dir, "tick_arith_mamba3_log.txt")


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
    decay_params = [p for n, p in param_dict.items() if p.ndim >= 2]
    no_decay_params = [p for n, p in param_dict.items() if p.ndim < 2]
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=max_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def masked_cross_entropy(logits, targets, mask):
    loss_per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)
    n_masked = mask.sum()
    if n_masked == 0:
        return torch.tensor(0.0, device=logits.device)
    return (loss_per_token * mask).sum() / n_masked


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
    prev_actions = torch.stack((
        prev_head0_moves,
        prev_head1_moves,
        prev_head0_writes,
        prev_head1_writes,
    ), dim=-1)
    targets = (
        target_head0_moves,
        target_head1_moves,
        target_head0_writes,
        target_head1_writes,
    )
    return reads, prev_actions, targets, loss_mask


def factorized_loss(logits, targets, mask):
    losses = [
        masked_cross_entropy(logit, target, mask)
        for logit, target in zip(logits, targets)
    ]
    return sum(losses) / len(losses)


def forward_recurrent(model, reads, prev_actions, targets, loss_mask):
    _, T = loss_mask.size()
    states = None
    latent = None
    total_loss = torch.tensor(0.0, device=loss_mask.device)
    total_masked = 0

    for pos in range(T):
        logits, latent, states = model(
            reads[:, pos:pos + 1].contiguous(),
            prev_actions[:, pos:pos + 1].contiguous(),
            latent=latent,
            states=states,
        )
        if DETACH_LATENT:
            latent = latent.detach()

        token_mask = loss_mask[:, pos:pos + 1].contiguous()
        n_masked = token_mask.sum().item()
        if n_masked > 0:
            token_targets = tuple(target[:, pos:pos + 1].contiguous() for target in targets)
            token_loss = factorized_loss(logits, token_targets, token_mask)
            total_loss = total_loss + token_loss * n_masked
            total_masked += n_masked

    return total_loss / total_masked if total_masked > 0 else total_loss


def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            reads, prev_actions, targets, loss_mask = unpack_batch(batch, DEVICE)
            _, T = loss_mask.size()
            states = None
            latent = None
            all_logits = [[] for _ in targets]

            for pos in range(T):
                logits, latent, states = model(
                    reads[:, pos:pos + 1].contiguous(),
                    prev_actions[:, pos:pos + 1].contiguous(),
                    latent=latent,
                    states=states,
                )
                if DETACH_LATENT:
                    latent = latent.detach()
                for i, logit in enumerate(logits):
                    all_logits[i].append(logit)

            all_logits = tuple(torch.cat(parts, dim=1) for parts in all_logits)
            batch_loss = factorized_loss(all_logits, targets, loss_mask)
            n_tokens = loss_mask.sum().item()
            total_loss += batch_loss.item() * n_tokens
            total_tokens += n_tokens

            factor_correct = torch.ones_like(loss_mask, dtype=torch.bool)
            for logit, target in zip(all_logits, targets):
                factor_correct &= logit.argmax(dim=-1).eq(target)
            correct += (factor_correct.float() * loss_mask).sum().item()
            total += n_tokens
    return total_loss / total_tokens, correct / total


def train():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    train_dataset = ArithmeticTickDataset(
        add_items=TRAIN_ADD,
        mul_items=TRAIN_MUL,
        tasks=TASKS,
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
        shuffle=True,
    )
    val_loader = ArithmeticTickDataLoader(
        val_dataset,
        batch_size=MICRO_BATCH_SIZE,
        pad_to_multiple=config.block_size,
        shuffle=False,
    )
    test_loader = ArithmeticTickDataLoader(
        test_dataset,
        batch_size=MICRO_BATCH_SIZE,
        pad_to_multiple=config.block_size,
        shuffle=False,
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
    print(f"Detach latent: {DETACH_LATENT}")
    print(f"Device: {DEVICE}")
    print(f"Train: {len(train_dataset)} examples")
    print(f"Total optimizer steps: {total_steps}")
    print()

    optimizer = configure_optimizer(model, weight_decay=0.1)
    with open(log_file, "w") as f:
        pass

    batch_iter = iter(train_loader)
    for step in range(total_steps):
        if step % EVAL_INTERVAL == 0:
            val_loss, val_acc = evaluate(raw_model, val_loader)
            print(f"step {step:4d}/{total_steps} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss:.4f} acc {val_acc:.4f}\n")

        lr = get_lr(step, total_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        t0 = time.time()

        for _ in range(GRAD_ACCUM_STEPS):
            batch = next(batch_iter)
            reads, prev_actions, targets, loss_mask = unpack_batch(batch, DEVICE)
            loss = forward_recurrent(model, reads, prev_actions, targets, loss_mask)
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            loss_accum += loss.detach()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        dt = time.time() - t0
        print(f"step {step + 1:4d}/{total_steps} | loss {loss_accum:.4f} | norm {norm:.4f} | lr {lr:.2e} | dt {dt:.2f}s")

    val_loss, val_acc = evaluate(raw_model, val_loader)
    test_loss, test_acc = evaluate(raw_model, test_loader)
    print(f"\nVal:         loss {val_loss:.4f} | acc {val_acc:.4f}")
    print(f"Test (long): loss {test_loss:.4f} | acc {test_acc:.4f}")
    with open(log_file, "a") as f:
        f.write(f"final val {val_loss:.4f} acc {val_acc:.4f}\n")
        f.write(f"final test {test_loss:.4f} acc {test_acc:.4f}\n")

    checkpoint_path = "log/tick_arith_mamba3_model.pt"
    torch.save({"model": raw_model.state_dict(), "config": config}, checkpoint_path)
    print(f"Saved model to {checkpoint_path}")


if __name__ == "__main__":
    train()
