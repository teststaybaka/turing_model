"""
Latent-feedback training script for the Turing machine memory tasks (recall / copy / mirror / repeat).

Single pass through training data (no epochs), step-based like FineWeb pretraining.
Uses masked cross-entropy on graded tokens only (recall: the recall WRITEs;
deferred branch: all action tokens).
Sequences are processed one token at a time so latent_out[t] feeds latent_in[t+1].
Gradient accumulation: effective_batch = micro_batch_size * grad_accum_steps.
"""

import os
import time
import math
import torch
import torch.nn.functional as F
from memory_tasks_data_loader import (
    TapeTaskDataset, TapeTaskDataLoader,
    TRAIN_STRINGS, VAL_STRINGS, TEST_LONG_STRINGS,
    TRAIN_RECALL, VAL_RECALL, TEST_LONG_RECALL, VOCAB_SIZE,
)
# Toggle model: "sliding", "stair", or "mamba3"
MODEL_TYPE = "mamba3"

if MODEL_TYPE == "sliding":
    from latent_rope_sliding_cache_model import GPT, GPTConfig
elif MODEL_TYPE == "stair":
    from latent_rope_stair_model import GPT, GPTConfig
elif MODEL_TYPE == "mamba3":
    from latent_mamba3_model import GPT, GPTConfig
else:
    raise ValueError(f"unknown MODEL_TYPE: {MODEL_TYPE}")

# --- Hyperparameters ---
MICRO_BATCH_SIZE = 128       # fits in GPU memory — adjust per hardware
GRAD_ACCUM_STEPS = 1        # effective batch 128 examples
EVAL_INTERVAL = 50          # eval every N optimizer steps
DEVICE = "cuda"
COMPILE = MODEL_TYPE == "mamba3"  # sliding/stair hit many cache-mask shapes token by token

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
log_file = os.path.join(log_dir, "latent_train_log.txt")


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


def forward_recurrent(model, input_ids, target_ids, loss_mask):
    """Process a sequence token by token with exact latent feedback."""
    B, T = input_ids.size()
    kv_caches = None
    latent = None
    total_loss = torch.tensor(0.0, device=input_ids.device)
    total_masked = 0

    for pos in range(T):
        token_inp = input_ids[:, pos:pos + 1].contiguous()
        token_tgt = target_ids[:, pos:pos + 1].contiguous()
        token_mask = loss_mask[:, pos:pos + 1].contiguous()

        logits, latent, kv_caches = model(token_inp, latent=latent, kv_caches=kv_caches)

        n_masked = token_mask.sum().item()
        if n_masked > 0:
            token_loss = masked_cross_entropy(logits, token_tgt, token_mask)
            total_loss = total_loss + token_loss * n_masked
            total_masked += n_masked

    return total_loss / total_masked if total_masked > 0 else total_loss


def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_actions = 0
    total_actions = 0
    with torch.no_grad():
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            for input_ids, target_ids, loss_mask in loader:
                input_ids = input_ids.to(DEVICE)
                target_ids = target_ids.to(DEVICE)
                loss_mask = loss_mask.to(DEVICE)

                B, T = input_ids.size()
                kv_caches = None
                latent = None
                all_logits = []

                for pos in range(T):
                    token = input_ids[:, pos:pos + 1].contiguous()
                    logits, latent, kv_caches = model(token, latent=latent, kv_caches=kv_caches)
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
    return total_loss / total_tokens, correct_actions / total_actions


def train():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision('high')

    chunk_size = config.block_size // 2 if MODEL_TYPE == "stair" else config.block_size

    train_dataset = TapeTaskDataset(TRAIN_STRINGS, TRAIN_RECALL)
    val_dataset = TapeTaskDataset(VAL_STRINGS, VAL_RECALL)
    test_dataset = TapeTaskDataset(TEST_LONG_STRINGS, TEST_LONG_RECALL)
    train_loader = TapeTaskDataLoader(train_dataset, batch_size=MICRO_BATCH_SIZE, chunk_size=chunk_size, shuffle=True)
    val_loader = TapeTaskDataLoader(val_dataset, batch_size=MICRO_BATCH_SIZE, chunk_size=chunk_size, shuffle=False)
    test_loader = TapeTaskDataLoader(test_dataset, batch_size=MICRO_BATCH_SIZE, chunk_size=chunk_size, shuffle=False)

    total_steps = len(train_loader) // GRAD_ACCUM_STEPS

    model = GPT(config).to(DEVICE)
    raw_model = model
    if COMPILE:
        model = torch.compile(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    print(f"Config: {config}")
    print(f"Chunk size: {chunk_size}")
    print(f"Compile: {COMPILE}")
    print(f"Device: {DEVICE}")
    print(f"Train: {len(train_dataset)} examples")
    print(f"Micro batch: {MICRO_BATCH_SIZE}, Grad accum: {GRAD_ACCUM_STEPS}, Effective batch: {MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS}")
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
            pg['lr'] = lr

        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0

        t0 = time.time()
        for micro_step in range(GRAD_ACCUM_STEPS):
            input_ids, target_ids, loss_mask = next(batch_iter)
            input_ids = input_ids.to(DEVICE)
            target_ids = target_ids.to(DEVICE)
            loss_mask = loss_mask.to(DEVICE)

            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                loss = forward_recurrent(model, input_ids, target_ids, loss_mask)

            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            loss_accum += loss.detach()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        dt = time.time() - t0
        print(f"step {step + 1:4d}/{total_steps} | loss {loss_accum:.4f} | norm {norm:.4f} | lr {lr:.2e} | dt {dt:.2f}s")

    # Final evaluation
    val_loss, val_acc = evaluate(raw_model, val_loader)
    test_loss, test_acc = evaluate(raw_model, test_loader)
    print(f"\nVal:         loss {val_loss:.4f} | acc {val_acc:.4f}")
    print(f"Test (long): loss {test_loss:.4f} | acc {test_acc:.4f}")
    with open(log_file, "a") as f:
        f.write(f"final val {val_loss:.4f} acc {val_acc:.4f}\n")
        f.write(f"final test {test_loss:.4f} acc {test_acc:.4f}\n")

    # Save the uncompiled module's weights so the checkpoint has clean keys
    # (no torch.compile "_orig_mod." prefix) and loads into a plain GPT.
    torch.save({
        'model': raw_model.state_dict(),
        'config': config,
    }, "log/latent_train_model.pt")
    print("Saved model to log/latent_train_model.pt")


if __name__ == "__main__":
    train()
