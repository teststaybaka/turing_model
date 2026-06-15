"""
Training script for the Turing machine copy task.

Single pass through training data (no epochs), step-based like FineWeb pretraining.
Uses masked cross-entropy on action tokens only.
Sequences split into chunks with KV cache carry and BPTT.
Gradient accumulation: effective_batch = micro_batch_size * grad_accum_steps.
"""

import os
import time
import math
import torch
import torch.nn.functional as F
from copy_task_data_loader import (
    CopyTaskDataset, CopyTaskDataLoader,
    TRAIN_STRINGS, VAL_STRINGS, TEST_LONG_STRINGS, VOCAB_SIZE,
)
# Toggle model: "sliding" or "stair"
MODEL_TYPE = "sliding"

if MODEL_TYPE == "sliding":
    from rope_sliding_cache_model import GPT, GPTConfig
elif MODEL_TYPE == "stair":
    from rope_stair_model import GPT, GPTConfig

# --- Hyperparameters ---
MICRO_BATCH_SIZE = 32       # fits in GPU memory — adjust per hardware
GRAD_ACCUM_STEPS = 4        # effective batch = 32 * 4 = 128 examples
EVAL_INTERVAL = 50          # eval every N optimizer steps
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

max_lr = 3e-4
min_lr = max_lr * 0.1
warmup_steps = 30

config = GPTConfig(
    block_size=32,
    vocab_size=VOCAB_SIZE,
    n_layers=4,
    n_heads=4,
    n_embd=64,
)

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")


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
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
    return torch.optim.AdamW(optim_groups, lr=max_lr, betas=(0.9, 0.95), eps=1e-8)


def masked_cross_entropy(logits, targets, mask):
    """Cross-entropy loss only on positions where mask == 1."""
    loss_per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction='none',
    ).view_as(targets)
    n_masked = mask.sum()
    if n_masked == 0:
        return torch.tensor(0.0, device=logits.device)
    return (loss_per_token * mask).sum() / n_masked


def forward_chunked(model, input_ids, target_ids, loss_mask, chunk_size):
    """Process a sequence in chunks with KV cache carry (BPTT — no detach)."""
    B, T = input_ids.size()
    kv_caches = None
    total_loss = torch.tensor(0.0, device=input_ids.device)
    total_masked = 0

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk_inp = input_ids[:, start:end]
        chunk_tgt = target_ids[:, start:end]
        chunk_mask = loss_mask[:, start:end]

        logits, kv_caches = model(chunk_inp, kv_caches=kv_caches)

        n_masked = chunk_mask.sum().item()
        if n_masked > 0:
            chunk_loss = masked_cross_entropy(logits, chunk_tgt, chunk_mask)
            total_loss = total_loss + chunk_loss * n_masked
            total_masked += n_masked

    return total_loss / total_masked if total_masked > 0 else total_loss


def evaluate(model, loader, chunk_size):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_actions = 0
    total_actions = 0
    with torch.no_grad():
        for input_ids, target_ids, loss_mask in loader:
            input_ids = input_ids.to(DEVICE)
            target_ids = target_ids.to(DEVICE)
            loss_mask = loss_mask.to(DEVICE)

            B, T = input_ids.size()
            kv_caches = None
            all_logits = []

            for start in range(0, T, chunk_size):
                end = min(start + chunk_size, T)
                logits, kv_caches = model(input_ids[:, start:end], kv_caches=kv_caches)
                all_logits.append(logits)

            all_logits = torch.cat(all_logits, dim=1)
            loss_per_token = F.cross_entropy(
                all_logits.reshape(-1, all_logits.size(-1)),
                target_ids.reshape(-1),
                reduction='none',
            ).view_as(target_ids)

            total_loss += (loss_per_token * loss_mask).sum().item()
            total_tokens += loss_mask.sum().item()

            preds = all_logits.argmax(dim=-1)
            correct_actions += ((preds == target_ids) * loss_mask).sum().item()
            total_actions += loss_mask.sum().item()

    model.train()
    return total_loss / total_tokens, correct_actions / total_actions


def train():
    train_dataset = CopyTaskDataset(TRAIN_STRINGS)
    val_dataset = CopyTaskDataset(VAL_STRINGS)
    test_dataset = CopyTaskDataset(TEST_LONG_STRINGS)
    train_loader = CopyTaskDataLoader(train_dataset, batch_size=MICRO_BATCH_SIZE, shuffle=True)
    val_loader = CopyTaskDataLoader(val_dataset, batch_size=MICRO_BATCH_SIZE, shuffle=False)
    test_loader = CopyTaskDataLoader(test_dataset, batch_size=MICRO_BATCH_SIZE, shuffle=False)

    chunk_size = config.block_size // 2 if MODEL_TYPE == "stair" else config.block_size
    total_steps = len(train_loader) // GRAD_ACCUM_STEPS

    model = GPT(config).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    print(f"Config: {config}")
    print(f"Chunk size: {chunk_size}")
    print(f"Device: {DEVICE}")
    print(f"Train: {len(train_dataset)} examples")
    print(f"Micro batch: {MICRO_BATCH_SIZE}, Grad accum: {GRAD_ACCUM_STEPS}, Effective batch: {MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"Total optimizer steps: {total_steps}")
    print()

    optimizer = configure_optimizer(model, weight_decay=0.1)

    with open(log_file, "w") as f:
        pass

    model.train()
    t0 = time.time()
    batch_iter = iter(train_loader)

    for step in range(total_steps):
        lr = get_lr(step, total_steps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(GRAD_ACCUM_STEPS):
            input_ids, target_ids, loss_mask = next(batch_iter)
            input_ids = input_ids.to(DEVICE)
            target_ids = target_ids.to(DEVICE)
            loss_mask = loss_mask.to(DEVICE)

            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                loss = forward_chunked(model, input_ids, target_ids, loss_mask, chunk_size)

            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            loss_accum += loss.item()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % EVAL_INTERVAL == 0 or step == total_steps - 1:
            dt = time.time() - t0
            val_loss, val_acc = evaluate(model, val_loader, chunk_size)
            print(f"step {step:4d}/{total_steps} | lr {lr:.2e} | loss {loss_accum:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | norm {norm:.4f} | {dt:.2f}s")
            with open(log_file, "a") as f:
                f.write(f"{step} loss {loss_accum:.4f} val {val_loss:.4f} acc {val_acc:.4f}\n")
            t0 = time.time()

    # Final evaluation
    val_loss, val_acc = evaluate(model, val_loader, chunk_size)
    test_loss, test_acc = evaluate(model, test_loader, chunk_size)
    print(f"\nVal:         loss {val_loss:.4f} | acc {val_acc:.4f}")
    print(f"Test (long): loss {test_loss:.4f} | acc {test_acc:.4f}")

    torch.save({
        'model': model.state_dict(),
        'config': config,
    }, "copy_task_model.pt")
    print("Saved model to copy_task_model.pt")


if __name__ == "__main__":
    train()
