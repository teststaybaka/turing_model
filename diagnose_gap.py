"""
Localize the reverse/mirror train-vs-val loss gap.

Single task, IID data, held-out streaming train loss => a real generalization
gap is essentially impossible. So the training-loss path (forward_chunked) and
the eval path (evaluate) must disagree on identical data. This script pins where.

  CHECK A — train() vs eval() at the LOGIT level, same batch, same chunked
            forward. These models have no dropout/BatchNorm, so the logits MUST
            be identical. If they differ, .train()/.eval() changes the forward
            (compile / FlexAttention) -> that is the bug.

  CHECK B — chunked+cache vs single-pass over the whole sequence (eval mode).
            If the cache/sliding path disagrees with a one-shot forward, the
            cache carry is wrong, and it compounds over reverse/mirror's extra
            chunks. (Only valid when padded_len <= block_size for the one-shot
            leg; we pick short sequences so the one-shot is a faithful reference.)

  CHECK C — the two scalar loss paths (forward_chunked vs evaluate) on the same
            batch, as a sanity number.

Run:  python3 diagnose_gap.py [checkpoint.pt]
Works with random weights — we are testing path equivalence, not quality.
Set DEVICE=cpu in train.py first if you have no GPU (also removes bf16 noise,
which makes any real disagreement unambiguous).
"""

import sys
import torch

from train import config, MODEL_TYPE, DEVICE
from tape_tasks_data_loader import (
    TapeTaskDataset, TapeTaskDataLoader, TRAIN_STRINGS,
)

if MODEL_TYPE == "sliding":
    from rope_sliding_cache_model import GPT
elif MODEL_TYPE == "stair":
    from rope_stair_model import GPT

TASK = "reverse"   # switch to "mirror" to check that one
CHUNK = config.block_size // 2 if MODEL_TYPE == "stair" else config.block_size


def chunked_logits(model, inp, chunk_size):
    """Replicate the train/eval chunked forward, returning concatenated logits."""
    T = inp.size(1)
    kv = None
    outs = []
    for start in range(0, T, chunk_size):
        chunk = inp[:, start:start + chunk_size].contiguous()
        logits, kv = model(chunk, kv_caches=kv)
        outs.append(logits)
    return torch.cat(outs, dim=1)


def main():
    autocast = dict(device_type=DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda"))
    model = GPT(config).to(DEVICE)
    model.precompute_block_masks(DEVICE)
    if len(sys.argv) > 1:
        ckpt = torch.load(sys.argv[1], map_location=DEVICE)
        sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
        model.load_state_dict(sd)
        print(f"Loaded {sys.argv[1]}")
    print(f"model={MODEL_TYPE} task={TASK} chunk={CHUNK} device={DEVICE}\n")

    # A long-ish batch so reverse spans many chunks (where the gap shows up).
    ds = TapeTaskDataset(TRAIN_STRINGS[:32], tasks=(TASK,))
    loader = TapeTaskDataLoader(ds, batch_size=32, chunk_size=CHUNK, shuffle=False)
    inp, tgt, mask = next(iter(loader))
    inp = inp.to(DEVICE)
    print(f"batch: B={inp.size(0)} T={inp.size(1)} ({inp.size(1)//CHUNK} chunks)\n")

    # ---- CHECK A: train() vs eval() logits ----
    with torch.no_grad(), torch.autocast(**autocast):
        model.train()
        lg_train = chunked_logits(model, inp, CHUNK)
        model.eval()
        lg_eval = chunked_logits(model, inp, CHUNK)
    dA = (lg_train.float() - lg_eval.float()).abs()
    print("CHECK A — train() vs eval() logits (must be ~0):")
    print(f"  max |diff| = {dA.max().item():.4e}   mean = {dA.mean().item():.4e}")
    if dA.max().item() > 1e-2:
        worst_t = dA.amax(dim=(0, 2)).argmax().item()
        print(f"  -> DIFFER. first big divergence near token pos {worst_t} "
              f"(chunk {worst_t // CHUNK}). .train()/.eval() changes the forward = BUG.")
    print()

    # ---- CHECK B: chunked+cache vs single-pass (short seqs only) ----
    short = TapeTaskDataset([s for s in TRAIN_STRINGS[:200] if len(s) <= CHUNK // 5][:32],
                            tasks=(TASK,))
    if len(short):
        sl = TapeTaskDataLoader(short, batch_size=len(short), chunk_size=CHUNK, shuffle=False)
        sinp, _, _ = next(iter(sl))
        sinp = sinp.to(DEVICE)
        with torch.no_grad(), torch.autocast(**autocast):
            model.eval()
            if sinp.size(1) <= config.block_size:
                full, _ = model(sinp, kv_caches=None)         # one shot, no carry
                chunked = chunked_logits(model, sinp, CHUNK)  # carried
                dB = (full.float() - chunked.float()).abs()
                print("CHECK B — single-pass vs chunked+cache (short seqs, must be ~0):")
                print(f"  T={sinp.size(1)}  max |diff| = {dB.max().item():.4e}  mean = {dB.mean().item():.4e}")
                if dB.max().item() > 1e-2:
                    print("  -> cache/chunk path disagrees with one-shot forward = BUG in cache carry.")
                print()


if __name__ == "__main__":
    main()
