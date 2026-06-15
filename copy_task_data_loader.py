"""
Copy task data loader for Turing machine training.

Trajectory format (example input "abc"):
  [COPY] [READ] [a] [WRITE_a] [READ] [b] [WRITE_b] [READ] [c] [WRITE_c] [READ] [BLANK] [DONE]

Loss is computed only on ACTION tokens (READ, WRITE_x, DONE).
Observation tokens ([a]-[h], [BLANK]) are masked out — the model isn't
expected to predict what the environment returns.

Heads auto-advance: each READ advances input head, each WRITE advances output head.
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

VOCAB_SIZE = 25

TOKEN_NAMES = {
    PAD: "[PAD]", COPY: "[COPY]", READ: "[READ]", BLANK: "[BLANK]", DONE: "[DONE]",
    MOVE_INPUT_LEFT: "[MOVE_INPUT_LEFT]", MOVE_INPUT_RIGHT: "[MOVE_INPUT_RIGHT]",
    MOVE_OUTPUT_LEFT: "[MOVE_OUTPUT_LEFT]", MOVE_OUTPUT_RIGHT: "[MOVE_OUTPUT_RIGHT]",
}
for i, c in enumerate(CHARS):
    TOKEN_NAMES[CHAR_BASE + i] = f"[{c}]"
    TOKEN_NAMES[WRITE_BASE + i] = f"[WRITE_{c}]"

ACTION_TOKENS = (
    {READ, DONE, MOVE_INPUT_LEFT, MOVE_INPUT_RIGHT, MOVE_OUTPUT_LEFT, MOVE_OUTPUT_RIGHT}
    | {WRITE_BASE + i for i in range(len(CHARS))}
)


def char_to_token(c):
    return CHAR_BASE + CHARS.index(c)


def char_to_write_token(c):
    return WRITE_BASE + CHARS.index(c)


def generate_trajectory(s):
    """Generate the full token sequence for copying string s.

    Trajectory for "abc":
      [COPY] [READ] [a] [WRITE_a] [MOVE_INPUT_RIGHT] [MOVE_OUTPUT_RIGHT]
             [READ] [b] [WRITE_b] [MOVE_INPUT_RIGHT] [MOVE_OUTPUT_RIGHT]
             [READ] [c] [WRITE_c] [MOVE_INPUT_RIGHT] [MOVE_OUTPUT_RIGHT]
             [READ] [BLANK] [DONE]
    """
    tokens = [COPY]
    for c in s:
        tokens.append(READ)
        tokens.append(char_to_token(c))
        tokens.append(char_to_write_token(c))
        tokens.append(MOVE_INPUT_RIGHT)
        tokens.append(MOVE_OUTPUT_RIGHT)
    tokens.append(READ)
    tokens.append(BLANK)
    tokens.append(DONE)
    return tokens




# --- Fixed dataset for reproducibility ---
def _generate_fixed_strings(n_examples, min_len, max_len, seed=42):
    rng = random.Random(seed)
    strings = []
    for _ in range(n_examples):
        length = rng.randint(min_len, max_len)
        s = "".join(rng.choice(CHARS) for _ in range(length))
        strings.append(s)
    return strings


TRAIN_STRINGS = _generate_fixed_strings(n_examples=10000, min_len=4, max_len=32, seed=42)
VAL_STRINGS = _generate_fixed_strings(n_examples=200, min_len=4, max_len=32, seed=123)
TEST_LONG_STRINGS = _generate_fixed_strings(n_examples=200, min_len=48, max_len=64, seed=456)


class CopyTaskDataset:
    def __init__(self, strings):
        self.trajectories = [generate_trajectory(s) for s in strings]

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]


class CopyTaskDataLoader:
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
        # Round up to multiple of chunk_size so every chunk the compiled
        # model sees is exactly chunk_size tokens — avoids recompilation.
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
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
            torch.tensor(loss_mask, dtype=torch.float32),
        )

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


if __name__ == "__main__":
    # Sanity check: print a few trajectories
    for s in TRAIN_STRINGS[:3]:
        tokens = generate_trajectory(s)
        print(f"Input: '{s}'")
        print(f"Tokens: {[TOKEN_NAMES[t] for t in tokens]}")
        print()

    # Test data loader
    dataset = CopyTaskDataset(TRAIN_STRINGS[:100])
    loader = CopyTaskDataLoader(dataset, batch_size=4, chunk_size=32)
    batch = next(iter(loader))
    input_ids, target_ids, loss_mask = batch
    print(f"Batch shapes: input={input_ids.shape}, target={target_ids.shape}, mask={loss_mask.shape}")
    print(f"Example input:  {[TOKEN_NAMES[t.item()] for t in input_ids[0] if t.item() != PAD]}")
    print(f"Example target: {[TOKEN_NAMES[t.item()] for t in target_ids[0] if t.item() != PAD]}")
    print(f"Example mask:   {[int(m.item()) for m in loss_mask[0][:20]]}")
