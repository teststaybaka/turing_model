"""
Memory-capability tape tasks for the Turing-machine LLM.

These tasks isolate ONE thing: can the architecture retain *bounded, constant*
information across an *unbounded distance* when that information is NOT re-emitted
and NOT spoon-fed back. That is the only capability the chunked-KV / relative-RoPE
"infinite context" claim actually rests on, and the one axis that separated stair
from sliding on the archived copy/mirror result.

Design rules (learned the hard way):
  - A *running accumulator* (an add carry, parity, a palindrome flag) is trivial:
    the honest trajectory updates it every step = re-emits it = local prediction.
    So none of those are here.
  - *Retrieval by head-scan* localizes too: walk the head there, READ, copy the
    spoon-fed observation. So the value we test is produced as a WRITE *action*
    from memory, with no scan-back and no second observation of it.
  - *Unbounded* memory (induction with an arbitrary late query, a Dyck stack) is
    just capacity / memorization in a finite KV cache. Not a circuit. Excluded.

What survives: hold a constant fact past a gap, consume it where the local context
is ambiguous. Two shapes:

  recall  (op RECALL) : k needle symbols, a SPLIT delimiter, then a length-G
                        haystack that never restates them; hitting the right BLANK
                        triggers writing every needle back out, in order, as WRITE
                        actions. k=1 is the classic needle (sweep G = retention
                        horizon); k>1 adds the capacity axis. All actions are
                        graded (a TM run must learn its own control flow); the
                        recall WRITEs are simply where retention is *measured* —
                        slice eval accuracy to those positions to read the horizon.

  deferred branch     : copy / mirror / repeat all share the IDENTICAL forward-copy
  (op COPY/MIRROR/     prefix and diverge only at the right boundary. The branch
       REPEAT)         decision (DONE vs. turn-around-and-reverse vs. rewind-and-
                        copy-again) is locally indistinguishable from the shared
                        prefix, so it can only be made by recalling the op token
                        ~5n tokens back. Generalizes "mirror beat sliding" to a
                        3-way branch at depth D = n. All actions graded.

Head conventions (explicit MOVE tokens):
  - input head READs the current input cell; off either end returns [BLANK].
  - WRITE_x writes the current output cell.
  - MOVE_INPUT_{LEFT,RIGHT} / MOVE_OUTPUT_{LEFT,RIGHT} step the heads one cell.
  - SPLIT is an on-tape delimiter (read like any cell) marking the end of the
    needles; it is an observation, never graded.
"""

import torch
import random

# --- Vocabulary ---
PAD = 0
READ = 1
BLANK = 2
DONE = 3
MOVE_INPUT_LEFT = 4
MOVE_INPUT_RIGHT = 5
MOVE_OUTPUT_LEFT = 6
MOVE_OUTPUT_RIGHT = 7
SPLIT = 8                 # on-tape delimiter between the needles and the filler
# op tokens (the prompt — never predicted)
RECALL = 9                # recall family (k=1 needle, k>1 k-needle)
COPY = 10                 # deferred-branch trio: all share the forward-copy prefix
MIRROR = 11
REPEAT = 12

SYMBOLS = list("abcdefgh")          # single alphabet for every task
CHAR_BASE = 13                      # observation tokens 13..20
WRITE_BASE = CHAR_BASE + len(SYMBOLS)   # write tokens 21..28

VOCAB_SIZE = WRITE_BASE + len(SYMBOLS)  # 29

TOKEN_NAMES = {
    PAD: "[PAD]", READ: "[READ]", BLANK: "[BLANK]", DONE: "[DONE]",
    MOVE_INPUT_LEFT: "[MOVE_INPUT_LEFT]", MOVE_INPUT_RIGHT: "[MOVE_INPUT_RIGHT]",
    MOVE_OUTPUT_LEFT: "[MOVE_OUTPUT_LEFT]", MOVE_OUTPUT_RIGHT: "[MOVE_OUTPUT_RIGHT]",
    SPLIT: "[SPLIT]",
    RECALL: "[RECALL]", COPY: "[COPY]", MIRROR: "[MIRROR]", REPEAT: "[REPEAT]",
}
for i, c in enumerate(SYMBOLS):
    TOKEN_NAMES[CHAR_BASE + i] = f"[{c}]"
    TOKEN_NAMES[WRITE_BASE + i] = f"[WRITE_{c}]"

OP_TOKENS = {RECALL, COPY, MIRROR, REPEAT}

# Tokens that count as model ACTIONS — loss is computed on these (every task).
# SPLIT/op/obs/BLANK/PAD are scaffold or observation and are never graded.
ACTION_TOKENS = (
    {READ, DONE, MOVE_INPUT_LEFT, MOVE_INPUT_RIGHT, MOVE_OUTPUT_LEFT, MOVE_OUTPUT_RIGHT}
    | {WRITE_BASE + i for i in range(len(SYMBOLS))}
)


def obs(sym):
    """Observation token for reading symbol `sym`."""
    return CHAR_BASE + SYMBOLS.index(sym)


def wrt(sym):
    """WRITE action token for symbol `sym`."""
    return WRITE_BASE + SYMBOLS.index(sym)


# --- Deferred-branch trio (shared forward-copy prefix) ------------------------
def generate_copy(s):
    """Forward copy, then DONE at the boundary."""
    toks = [COPY]
    for c in s:
        toks += [READ, obs(c), wrt(c), MOVE_INPUT_RIGHT, MOVE_OUTPUT_RIGHT]
    toks += [READ, BLANK, DONE]
    return toks


def generate_mirror(s):
    """Forward copy (shared prefix), then turn around and append reverse(s)."""
    toks = [MIRROR]
    for c in s:                                  # ---- shared forward-copy prefix ----
        toks += [READ, obs(c), wrt(c), MOVE_INPUT_RIGHT, MOVE_OUTPUT_RIGHT]
    toks += [READ, BLANK, MOVE_INPUT_LEFT]       # boundary: turn (no write)
    for c in reversed(s):                        # backward read / forward write
        toks += [READ, obs(c), wrt(c), MOVE_INPUT_LEFT, MOVE_OUTPUT_RIGHT]
    toks += [READ, BLANK, DONE]
    return toks


def generate_repeat(s):
    """Forward copy (shared prefix), rewind the input head to the start, copy
    forward a second time -> output s + s."""
    toks = [REPEAT]
    for c in s:                                  # ---- shared forward-copy prefix ----
        toks += [READ, obs(c), wrt(c), MOVE_INPUT_RIGHT, MOVE_OUTPUT_RIGHT]
    toks += [READ, BLANK]                         # boundary
    for c in reversed(s):                        # rewind: step left, read each cell
        toks += [MOVE_INPUT_LEFT, READ, obs(c)]
    toks += [MOVE_INPUT_LEFT, READ, BLANK]        # off the left end
    toks += [MOVE_INPUT_RIGHT]                     # back to index 0
    for c in s:                                  # second forward copy
        toks += [READ, obs(c), wrt(c), MOVE_INPUT_RIGHT, MOVE_OUTPUT_RIGHT]
    toks += [READ, BLANK, DONE]
    return toks


# --- Recall family (needle / k-needle) ---------------------------------------
def generate_recall(needles, filler):
    """Read k `needles`, a SPLIT delimiter, then a length-G `filler` haystack that
    never restates them; on hitting the right BLANK, write every needle back out
    in order.

    The needle values are never re-observed at recall time, so a correct recall
    WRITE can only come from memory held across G — exactly what a bounded window
    must drop. Those WRITE positions (recall's only WRITEs) are where the retention
    horizon is read off at eval time.
    """
    toks = [RECALL]
    for v in needles:                            # needles to remember (before SPLIT)
        toks += [READ, obs(v), MOVE_INPUT_RIGHT]
    toks += [READ, SPLIT, MOVE_INPUT_RIGHT]       # delimiter: the needles end here
    for c in filler:                             # haystack: read across the gap
        toks += [READ, obs(c), MOVE_INPUT_RIGHT]
    toks += [READ, BLANK]                         # off the right end -> recall trigger
    for v in needles:                            # recall: write each needle from memory
        toks += [wrt(v), MOVE_OUTPUT_RIGHT]
    toks += [DONE]
    return toks


BRANCH_TASKS = {
    "copy": generate_copy,
    "mirror": generate_mirror,
    "repeat": generate_repeat,
}
DEFAULT_TASKS = ["recall", "copy", "mirror", "repeat"]


# --- Fixed dataset for reproducibility ---------------------------------------
def _gen_strings(n_examples, min_len, max_len, seed):
    rng = random.Random(seed)
    return ["".join(rng.choice(SYMBOLS) for _ in range(rng.randint(min_len, max_len)))
            for _ in range(n_examples)]


TRAIN_STRINGS = _gen_strings(100000, 4, 32, seed=42)
VAL_STRINGS = _gen_strings(200, 4, 32, seed=123)
TEST_LONG_STRINGS = _gen_strings(200, 48, 64, seed=456)


def _gen_recall_items(n_examples, min_k, max_k, min_g, max_g, seed):
    """Random (needles, filler): k needles in [min_k,max_k], gap G in [min_g,max_g]."""
    rng = random.Random(seed)
    items = []
    for _ in range(n_examples):
        k = rng.randint(min_k, max_k)
        g = rng.randint(min_g, max_g)
        needles = [rng.choice(SYMBOLS) for _ in range(k)]
        filler = [rng.choice(SYMBOLS) for _ in range(g)]
        items.append((needles, filler))
    return items


# recall: k in 1..4 (k=1 needle + k>1 k-needle), gap G matched to the string lengths
# so distance scales the same way; the long test set pushes G to 48..64.
TRAIN_RECALL = _gen_recall_items(100000, 1, 4, 4, 32, seed=42)
VAL_RECALL = _gen_recall_items(200, 1, 4, 4, 32, seed=123)
TEST_LONG_RECALL = _gen_recall_items(200, 1, 4, 48, 64, seed=456)


class TapeTaskDataset:
    """One trajectory per (item, task). Branch tasks consume `strings`; the recall
    task consumes `recall_items`. Each item is a token list (one TM run)."""

    def __init__(self, strings, recall_items, tasks=DEFAULT_TASKS):
        self.tasks = tuple(tasks)
        data = []
        branch = [t for t in self.tasks if t in BRANCH_TASKS]
        for s in strings:
            for t in branch:
                data.append(BRANCH_TASKS[t](s))
        if "recall" in self.tasks:
            for needles, filler in recall_items:
                data.append(generate_recall(needles, filler))
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TapeTaskDataLoader:
    def __init__(self, dataset, batch_size, chunk_size, shuffle=True, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.rng = random.Random(seed)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            self.rng.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch = [self.dataset[i] for i in indices[start:start + self.batch_size]]
            yield self._collate(batch)

    def _collate(self, batch):
        # Round up to a multiple of chunk_size so every chunk is exactly
        # chunk_size tokens — avoids recompilation.
        seq_len = max(len(toks) - 1 for toks in batch)
        padded_len = ((seq_len + self.chunk_size - 1) // self.chunk_size) * self.chunk_size
        input_ids, target_ids, loss_mask = [], [], []
        for toks in batch:
            inp, tgt = toks[:-1], toks[1:]
            mask = [1 if t in ACTION_TOKENS else 0 for t in tgt]   # loss on action tokens only
            pad_len = padded_len - len(inp)
            input_ids.append(inp + [PAD] * pad_len)
            target_ids.append(tgt + [PAD] * pad_len)
            loss_mask.append(mask + [0] * pad_len)

        # Pad the BATCH dimension up to batch_size — pad rows get an all-zero mask.
        while len(input_ids) < self.batch_size:
            input_ids.append([PAD] * padded_len)
            target_ids.append([PAD] * padded_len)
            loss_mask.append([0] * padded_len)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
            torch.tensor(loss_mask, dtype=torch.float32),
        )

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


if __name__ == "__main__":
    def decode_output(toks):
        return "".join(SYMBOLS[t - WRITE_BASE] for t in toks
                       if WRITE_BASE <= t < WRITE_BASE + len(SYMBOLS))

    def action_mask(toks):
        return [1 if t in ACTION_TOKENS else 0 for t in toks]

    print("== deferred branch (output) ==")
    for name, gen in BRANCH_TASKS.items():
        toks = gen("abc")
        print(f"  {name:7s} 'abc' -> '{decode_output(toks)}'")

    # shared forward-copy prefix: copy/mirror/repeat must agree up to the boundary
    s = "abcd"
    cp = generate_copy(s)
    mr = generate_mirror(s)
    rp = generate_repeat(s)
    # compare the bodies (index 1 onward); the op token at index 0 differs by design
    div = 1
    while div < min(len(cp), len(mr), len(rp)) and cp[div] == mr[div] == rp[div]:
        div += 1
    forward = 1 + 5 * len(s) + 2     # op + (READ obs wrt MOVE MOVE)*n + READ BLANK
    print(f"  identical body prefix of copy/mirror/repeat = {div} tokens "
          f"(forward-copy boundary at {forward}) -> branch deferred to depth ~{5*len(s)}")
    assert div == forward, (div, forward)
    assert cp[div] != mr[div] or mr[div] != rp[div], "branch must diverge at the boundary"

    print("== recall (needle k=1 and k-needle k>1) ==")
    for needles in (["c"], ["a", "f", "b"]):
        filler = list("hghghghg")
        toks = generate_recall(needles, filler)
        grade = action_mask(toks)
        recalled = decode_output(toks)
        writes = [t for t, g in zip(toks, grade) if g and WRITE_BASE <= t < WRITE_BASE + len(SYMBOLS)]
        assert recalled == "".join(needles), (recalled, needles)
        assert writes == [wrt(v) for v in needles], "recall's only WRITEs are the needles"
        print(f"  k={len(needles)} needles={needles} gap={len(filler)} "
              f"-> recalled '{recalled}'  (graded actions: {sum(grade)}, recall-writes: {len(writes)})")

    ds = TapeTaskDataset(TRAIN_STRINGS[:50], TRAIN_RECALL[:50])
    print(f"\nVOCAB_SIZE={VOCAB_SIZE}  dataset={len(ds)} trajectories  tasks={DEFAULT_TASKS}")
