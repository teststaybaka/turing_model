"""
Tape-task data loader for Turing-machine training.

Three string tasks, selected by the leading op token, sharing one head/MOVE
vocabulary. The op token sits at position 0 and is never a target (it is the
prompt), but it governs the action at every step — so a model trained on the
mix must carry that op token across the whole trajectory.

  copy     "abc" -> "abc"
  reverse  "abc" -> "cba"
  mirror   "abc" -> "abccba"   (s followed by reverse(s))

All trajectories are honest TM runs over a re-readable tape: the head moves one
cell at a time and BLANK is observed at either boundary. Loss is computed only
on ACTION tokens (READ, WRITE_x, MOVE_*, DONE). Observation tokens ([a]-[h],
[BLANK]) are masked out — the model isn't expected to predict what the tape
returns. The op token is masked out too (it is given, not predicted).

Head conventions (explicit MOVE tokens, both directions used):
  - input head READs the current input cell; off either end returns [BLANK].
  - WRITE_x writes the current output cell.
  - MOVE_INPUT_{LEFT,RIGHT} / MOVE_OUTPUT_RIGHT step the heads one cell.
"""

import torch
import random

# --- Vocabulary ---
CHARS = list("abcdefgh")

PAD = 0
COPY = 1
READ = 2
BLANK = 3
DONE = 4
MOVE_INPUT_LEFT = 5
MOVE_INPUT_RIGHT = 6
MOVE_OUTPUT_LEFT = 7
MOVE_OUTPUT_RIGHT = 8
CHAR_BASE = 9          # a=9, b=10, ..., h=16
WRITE_BASE = 17        # WRITE_a=17, WRITE_b=18, ..., WRITE_h=24
REVERSE = 25           # op token (new)
MIRROR = 26            # op token (new)

VOCAB_SIZE = 27

TOKEN_NAMES = {
    PAD: "[PAD]", COPY: "[COPY]", READ: "[READ]", BLANK: "[BLANK]", DONE: "[DONE]",
    MOVE_INPUT_LEFT: "[MOVE_INPUT_LEFT]", MOVE_INPUT_RIGHT: "[MOVE_INPUT_RIGHT]",
    MOVE_OUTPUT_LEFT: "[MOVE_OUTPUT_LEFT]", MOVE_OUTPUT_RIGHT: "[MOVE_OUTPUT_RIGHT]",
    REVERSE: "[REVERSE]", MIRROR: "[MIRROR]",
}
for i, c in enumerate(CHARS):
    TOKEN_NAMES[CHAR_BASE + i] = f"[{c}]"
    TOKEN_NAMES[WRITE_BASE + i] = f"[WRITE_{c}]"

# Op tokens are the prompt — never predicted, so not in ACTION_TOKENS.
OP_TOKENS = {COPY, REVERSE, MIRROR}

ACTION_TOKENS = (
    {READ, DONE, MOVE_INPUT_LEFT, MOVE_INPUT_RIGHT, MOVE_OUTPUT_LEFT, MOVE_OUTPUT_RIGHT}
    | {WRITE_BASE + i for i in range(len(CHARS))}
)


def char_to_token(c):
    return CHAR_BASE + CHARS.index(c)


def char_to_write_token(c):
    return WRITE_BASE + CHARS.index(c)


# --- Trajectory generators (one honest TM run each) ---
def generate_copy(s):
    """Read+write left to right, both heads advance in lockstep.

    "abc": [COPY] (READ a WRITE_a MIR MOR)x3  READ BLANK DONE
    """
    tokens = [COPY]
    for c in s:
        tokens += [READ, char_to_token(c), char_to_write_token(c),
                   MOVE_INPUT_RIGHT, MOVE_OUTPUT_RIGHT]
    tokens += [READ, BLANK, DONE]
    return tokens


def generate_reverse(s):
    """Scan input head to the right boundary, turn around, then read backward
    while writing forward.

    "abc": [REVERSE]
           (READ a MIR)(READ b MIR)(READ c MIR)        # scan to end
           READ BLANK MIL                              # right boundary, turn
           (READ c WRITE_c MIL MOR) ... (READ a ...)   # emit reversed
           READ BLANK DONE                             # left boundary
    """
    tokens = [REVERSE]
    for c in s:                                  # scan right to the boundary
        tokens += [READ, char_to_token(c), MOVE_INPUT_RIGHT]
    tokens += [READ, BLANK, MOVE_INPUT_LEFT]     # hit right BLANK, turn around
    for c in reversed(s):                        # read backward, write forward
        tokens += [READ, char_to_token(c), char_to_write_token(c),
                   MOVE_INPUT_LEFT, MOVE_OUTPUT_RIGHT]
    tokens += [READ, BLANK, DONE]                # hit left BLANK
    return tokens


def generate_mirror(s):
    """Copy forward (writing as we go), hit the right boundary, turn around,
    then read backward while still writing forward. Output head only moves
    right, so the second half lands contiguously after the first.

    "abc" -> output "abccba".
    """
    tokens = [MIRROR]
    for c in s:                                  # forward copy phase
        tokens += [READ, char_to_token(c), char_to_write_token(c),
                   MOVE_INPUT_RIGHT, MOVE_OUTPUT_RIGHT]
    tokens += [READ, BLANK, MOVE_INPUT_LEFT]     # right boundary, turn (no write)
    for c in reversed(s):                        # backward mirror phase
        tokens += [READ, char_to_token(c), char_to_write_token(c),
                   MOVE_INPUT_LEFT, MOVE_OUTPUT_RIGHT]
    tokens += [READ, BLANK, DONE]                # left boundary
    return tokens


TASKS = {
    "copy": generate_copy,
    "reverse": generate_reverse,
    "mirror": generate_mirror,
}
DEFAULT_TASKS = ("copy", "reverse", "mirror")


# --- Fixed dataset for reproducibility ---
def _generate_fixed_strings(n_examples, min_len, max_len, seed=42):
    rng = random.Random(seed)
    strings = []
    for _ in range(n_examples):
        length = rng.randint(min_len, max_len)
        s = "".join(rng.choice(CHARS) for _ in range(length))
        strings.append(s)
    return strings


TRAIN_STRINGS = _generate_fixed_strings(n_examples=100000, min_len=4, max_len=32, seed=42)
VAL_STRINGS = _generate_fixed_strings(n_examples=200, min_len=4, max_len=32, seed=123)
TEST_LONG_STRINGS = _generate_fixed_strings(n_examples=200, min_len=48, max_len=64, seed=456)


class TapeTaskDataset:
    """Builds one trajectory per (string, task) so the three tasks are balanced
    and the model must read the op token to know what to do."""

    def __init__(self, strings, tasks=DEFAULT_TASKS):
        self.tasks = tuple(tasks)
        self.trajectories = [TASKS[t](s) for s in strings for t in self.tasks]

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]


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
            batch_indices = indices[start:start + self.batch_size]
            batch = [self.dataset[i] for i in batch_indices]
            yield self._collate(batch)

    def _collate(self, batch):
        # Round up to a multiple of chunk_size so every chunk the compiled model
        # sees is exactly chunk_size tokens — avoids recompilation.
        seq_len = max(len(tokens) - 1 for tokens in batch)
        padded_len = ((seq_len + self.chunk_size - 1) // self.chunk_size) * self.chunk_size
        input_ids = []
        target_ids = []
        loss_mask = []
        for tokens in batch:
            inp = tokens[:-1]
            tgt = tokens[1:]
            mask = [1 if t in ACTION_TOKENS else 0 for t in tgt]
            pad_len = padded_len - len(inp)
            input_ids.append(inp + [PAD] * pad_len)
            target_ids.append(tgt + [PAD] * pad_len)
            loss_mask.append(mask + [0] * pad_len)

        # Pad the BATCH dimension up to batch_size so the compiled model always
        # sees exactly batch_size rows — pad rows get an all-zero loss mask.
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
    # Sanity check: print one trajectory per task and verify the output stream.
    def decode_output(tokens):
        out = []
        for t in tokens:
            if WRITE_BASE <= t < WRITE_BASE + len(CHARS):
                out.append(CHARS[t - WRITE_BASE])
        return "".join(out)

    s = "abc"
    for name, gen in TASKS.items():
        tokens = gen(s)
        print(f"{name:8s} '{s}' -> output '{decode_output(tokens)}'")
        print("  " + " ".join(TOKEN_NAMES[t] for t in tokens))
        print()

    dataset = TapeTaskDataset(TRAIN_STRINGS[:100])
    loader = TapeTaskDataLoader(dataset, batch_size=4, chunk_size=16)
    input_ids, target_ids, loss_mask = next(iter(loader))
    print(f"Batch shapes: input={tuple(input_ids.shape)}, "
          f"target={tuple(target_ids.shape)}, mask={tuple(loss_mask.shape)}")
    print(f"Dataset size: {len(dataset)} trajectories "
          f"({len(TRAIN_STRINGS[:100])} strings x {len(DEFAULT_TASKS)} tasks)")
