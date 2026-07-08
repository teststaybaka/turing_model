"""Supervised tick-level Turing-machine training with Mamba-3.

There is no READ action. At each tick the model observes both tape heads, the
previous action, and the previous latent output, then predicts the next action.
"""

import os
import time
import math
import torch
import torch.nn.functional as F

from tick_tape_data_loader import (
    TickTaskDataset,
    TickTaskDataLoader,
    TRAIN_STRINGS,
    VAL_STRINGS,
    TEST_LONG_STRINGS,
    READ_VOCAB_SIZE,
    ACTION_VOCAB_SIZE,
)
from tick_mamba3_model import GPT, GPTConfig


MICRO_BATCH_SIZE = 128
GRAD_ACCUM_STEPS = 1
EVAL_INTERVAL = 50
DEVICE = "cuda"
COMPILE = True
DETACH_LATENT = False
TASKS = ["copy", "mirror", "repeat"]

max_lr = 3e-4
min_lr = max_lr * 0.1
warmup_steps = 30

config = GPTConfig(
    block_size=32,
    read_vocab_size=READ_VOCAB_SIZE,
    action_vocab_size=ACTION_VOCAB_SIZE,
    n_layers=4,
    n_heads=4,
    n_embd=64,
)

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "tick_mamba3_log.txt")


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


def forward_recurrent(model, input_reads, output_reads, prev_actions, target_actions, loss_mask):
    _, T = target_actions.size()
    states = None
    latent = None
    total_loss = torch.tensor(0.0, device=target_actions.device)
    total_masked = 0

    for pos in range(T):
        logits, latent, states = model(
            input_reads[:, pos:pos + 1].contiguous(),
            output_reads[:, pos:pos + 1].contiguous(),
            prev_actions[:, pos:pos + 1].contiguous(),
            latent=latent,
            states=states,
        )
        if DETACH_LATENT:
            latent = latent.detach()

        token_mask = loss_mask[:, pos:pos + 1].contiguous()
        n_masked = token_mask.sum().item()
        if n_masked > 0:
            token_loss = masked_cross_entropy(
                logits,
                target_actions[:, pos:pos + 1].contiguous(),
                token_mask,
            )
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
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            for batch in loader:
                input_reads, output_reads, prev_actions, target_actions, loss_mask = [
                    x.to(DEVICE) for x in batch
                ]
                _, T = target_actions.size()
                states = None
                latent = None
                all_logits = []

                for pos in range(T):
                    logits, latent, states = model(
                        input_reads[:, pos:pos + 1].contiguous(),
                        output_reads[:, pos:pos + 1].contiguous(),
                        prev_actions[:, pos:pos + 1].contiguous(),
                        latent=latent,
                        states=states,
                    )
                    if DETACH_LATENT:
                        latent = latent.detach()
                    all_logits.append(logits)

                all_logits = torch.cat(all_logits, dim=1)
                loss_per_token = F.cross_entropy(
                    all_logits.reshape(-1, all_logits.size(-1)),
                    target_actions.reshape(-1),
                    reduction="none",
                ).view_as(target_actions)
                total_loss += (loss_per_token * loss_mask).sum().item()
                total_tokens += loss_mask.sum().item()

                preds = all_logits.argmax(dim=-1)
                correct += ((preds == target_actions) * loss_mask).sum().item()
                total += loss_mask.sum().item()
    return total_loss / total_tokens, correct / total


def train():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    train_dataset = TickTaskDataset(TRAIN_STRINGS, tasks=TASKS)
    val_dataset = TickTaskDataset(VAL_STRINGS, tasks=TASKS)
    test_dataset = TickTaskDataset(TEST_LONG_STRINGS, tasks=TASKS)

    train_loader = TickTaskDataLoader(train_dataset, batch_size=MICRO_BATCH_SIZE, pad_to_multiple=config.block_size, shuffle=True)
    val_loader = TickTaskDataLoader(val_dataset, batch_size=MICRO_BATCH_SIZE, pad_to_multiple=config.block_size, shuffle=False)
    test_loader = TickTaskDataLoader(test_dataset, batch_size=MICRO_BATCH_SIZE, pad_to_multiple=config.block_size, shuffle=False)
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
            input_reads, output_reads, prev_actions, target_actions, loss_mask = [
                x.to(DEVICE) for x in batch
            ]
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                loss = forward_recurrent(model, input_reads, output_reads, prev_actions, target_actions, loss_mask)
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

    torch.save({"model": raw_model.state_dict(), "config": config}, "log/tick_mamba3_model.pt")
    print("Saved model to log/tick_mamba3_model.pt")


if __name__ == "__main__":
    train()
