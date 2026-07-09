"""Two-head, one-tape tick-level tasks.

Each training row is one machine tick. There is no READ action: both heads read
from the same tape on every tick. The supervised target is factorized into
per-head moves and per-head writes. There is no explicit done flag: the sequence
simply ends (the last tick is a STAY/NOOP) and everything after is padding.

Initial tape layout for copy/mirror/reverse:
  [TASK] chars... [END] blanks...

Initial tape layout for recall:
  [RECALL] chars... [SEP] gap [END] blanks...

Head 0 is the source head. Head 1 is the destination head. The appended output
starts at the first blank cell after the initial tape.
"""

from dataclasses import dataclass
import random


# --- Tape/read vocabulary ----------------------------------------------------
READ_PAD = 0
READ_BLANK = 1
READ_COPY = 2
READ_MIRROR = 3
READ_RECALL = 4
READ_REVERSE = 5
READ_SEP = 6
READ_END = 7

SYMBOLS = list("abcdefgh")
READ_CHAR_BASE = 8
READ_VOCAB_SIZE = READ_CHAR_BASE + len(SYMBOLS)


# --- Move target vocabulary --------------------------------------------------
MOVE_PAD = 0
MOVE_START = 1
MOVE_STAY = 2
MOVE_LEFT = 3
MOVE_RIGHT = 4
MOVE_VOCAB_SIZE = 5


# --- Write target vocabulary -------------------------------------------------
WRITE_PAD = 0
WRITE_START = 1
WRITE_NOOP = 2
WRITE_BLANK = 3
WRITE_COPY = 4
WRITE_MIRROR = 5
WRITE_RECALL = 6
WRITE_REVERSE = 7
WRITE_SEP = 8
WRITE_END = 9
WRITE_CHAR_BASE = 10
WRITE_VOCAB_SIZE = WRITE_CHAR_BASE + len(SYMBOLS)


READ_TOKEN_NAMES = {
    READ_PAD: "[PAD]",
    READ_BLANK: "[BLANK]",
    READ_COPY: "[COPY]",
    READ_MIRROR: "[MIRROR]",
    READ_RECALL: "[RECALL]",
    READ_REVERSE: "[REVERSE]",
    READ_SEP: "[SEP]",
    READ_END: "[END]",
}
MOVE_TOKEN_NAMES = {
    MOVE_PAD: "[PAD]",
    MOVE_START: "[START]",
    MOVE_STAY: "[STAY]",
    MOVE_LEFT: "[LEFT]",
    MOVE_RIGHT: "[RIGHT]",
}
WRITE_TOKEN_NAMES = {
    WRITE_PAD: "[PAD]",
    WRITE_START: "[START]",
    WRITE_NOOP: "[NOOP]",
    WRITE_BLANK: "[WRITE_BLANK]",
    WRITE_COPY: "[WRITE_COPY]",
    WRITE_MIRROR: "[WRITE_MIRROR]",
    WRITE_RECALL: "[WRITE_RECALL]",
    WRITE_REVERSE: "[WRITE_REVERSE]",
    WRITE_SEP: "[WRITE_SEP]",
    WRITE_END: "[WRITE_END]",
}
for i, c in enumerate(SYMBOLS):
    READ_TOKEN_NAMES[READ_CHAR_BASE + i] = f"[{c}]"
    WRITE_TOKEN_NAMES[WRITE_CHAR_BASE + i] = f"[WRITE_{c}]"


def read_char(sym):
    return READ_CHAR_BASE + SYMBOLS.index(sym)


def write_char(sym):
    return WRITE_CHAR_BASE + SYMBOLS.index(sym)


def write_to_read(write):
    if write == WRITE_NOOP:
        return None
    if write == WRITE_BLANK:
        return READ_BLANK
    if write == WRITE_COPY:
        return READ_COPY
    if write == WRITE_MIRROR:
        return READ_MIRROR
    if write == WRITE_RECALL:
        return READ_RECALL
    if write == WRITE_REVERSE:
        return READ_REVERSE
    if write == WRITE_SEP:
        return READ_SEP
    if write == WRITE_END:
        return READ_END
    if WRITE_CHAR_BASE <= write < WRITE_VOCAB_SIZE:
        return READ_CHAR_BASE + (write - WRITE_CHAR_BASE)
    raise ValueError(f"invalid write token {write}")


@dataclass
class TickTrajectory:
    task: str
    head0_reads: list[int]
    head1_reads: list[int]
    target_head0_moves: list[int]
    target_head1_moves: list[int]
    target_head0_writes: list[int]
    target_head1_writes: list[int]
    final_output: str

    @property
    def prev_head0_moves(self):
        return [MOVE_START] + self.target_head0_moves[:-1]

    @property
    def prev_head1_moves(self):
        return [MOVE_START] + self.target_head1_moves[:-1]

    @property
    def prev_head0_writes(self):
        return [WRITE_START] + self.target_head0_writes[:-1]

    @property
    def prev_head1_writes(self):
        return [WRITE_START] + self.target_head1_writes[:-1]


class _Program:
    def __init__(self):
        self.head0 = 0
        self.head1 = 0
        self.head0_moves = []
        self.head1_moves = []
        self.head0_writes = []
        self.head1_writes = []

    def step(
        self,
        head0_move=MOVE_STAY,
        head1_move=MOVE_STAY,
        head0_write=WRITE_NOOP,
        head1_write=WRITE_NOOP,
    ):
        self.head0_moves.append(head0_move)
        self.head1_moves.append(head1_move)
        self.head0_writes.append(head0_write)
        self.head1_writes.append(head1_write)
        self.head0 += _move_delta(head0_move)
        self.head1 += _move_delta(head1_move)

    def move_heads_to(self, head0_target, head1_target):
        while self.head0 != head0_target or self.head1 != head1_target:
            self.step(
                head0_move=_move_toward(self.head0, head0_target),
                head1_move=_move_toward(self.head1, head1_target),
            )

    def finish(self):
        # Terminal no-op tick: both heads STAY and NOOP. No done flag — the
        # sequence just ends here and everything after is padding.
        self.step()


def _move_delta(move):
    if move == MOVE_LEFT:
        return -1
    if move == MOVE_RIGHT:
        return 1
    if move == MOVE_STAY:
        return 0
    raise ValueError(f"invalid move token {move}")


def _move_toward(pos, target):
    if pos < target:
        return MOVE_RIGHT
    if pos > target:
        return MOVE_LEFT
    return MOVE_STAY


def _read_tape(tape, pos):
    return tape.get(pos, READ_BLANK)


def _apply_write(tape, pos, write):
    token = write_to_read(write)
    if token is None:
        return
    tape[pos] = token


def _initial_tape(task_token, s):
    return [task_token] + [read_char(c) for c in s] + [READ_END]


def _simulate(task, initial_tape, program, expected_output, output_start=None):
    tape = {i: token for i, token in enumerate(initial_tape)}
    head0 = 0
    head1 = 0
    head0_reads = []
    head1_reads = []

    n = len(program.head0_moves)
    for i in range(n):
        head0_reads.append(_read_tape(tape, head0))
        head1_reads.append(_read_tape(tape, head1))

        _apply_write(tape, head0, program.head0_writes[i])
        _apply_write(tape, head1, program.head1_writes[i])

        head0 += _move_delta(program.head0_moves[i])
        head1 += _move_delta(program.head1_moves[i])

    if output_start is None:
        output_start = len(initial_tape)
    final_output = []
    for i in range(len(expected_output)):
        token = _read_tape(tape, output_start + i)
        expected = read_char(expected_output[i])
        if token != expected:
            raise AssertionError(f"{task}: output[{i}] expected {expected}, got {token}")
        final_output.append(expected_output[i])

    return TickTrajectory(
        task=task,
        head0_reads=head0_reads,
        head1_reads=head1_reads,
        target_head0_moves=list(program.head0_moves),
        target_head1_moves=list(program.head1_moves),
        target_head0_writes=list(program.head0_writes),
        target_head1_writes=list(program.head1_writes),
        final_output="".join(final_output),
    )


def generate_copy(s):
    initial_tape = _initial_tape(READ_COPY, s)
    output_start = len(initial_tape)
    program = _Program()
    program.move_heads_to(1, output_start)
    for c in s:
        program.step(
            head0_move=MOVE_RIGHT,
            head1_move=MOVE_RIGHT,
            head1_write=write_char(c),
        )
    program.finish()
    return _simulate("copy", initial_tape, program, expected_output=s)


def generate_mirror(s):
    initial_tape = _initial_tape(READ_MIRROR, s)
    output_start = len(initial_tape)
    program = _Program()

    # Head 0 discovers the source boundary by reading [END], then backs up to
    # the last source character. Head 1 aligns to the
    # append region in parallel.
    program.move_heads_to(len(s) + 1, output_start)
    program.step(head0_move=MOVE_LEFT)

    for c in reversed(s):
        program.step(
            head0_move=MOVE_LEFT,
            head1_move=MOVE_RIGHT,
            head1_write=write_char(c),
        )
    program.finish()
    return _simulate("mirror", initial_tape, program, expected_output=s[::-1])


def _recall_gap(s):
    gap_len = max(32, 4 * len(s))
    return "".join(SYMBOLS[(i + len(s)) % len(SYMBOLS)] for i in range(gap_len))


def generate_recall(s):
    gap = _recall_gap(s)
    initial_tape = [READ_RECALL] + [read_char(c) for c in s] + [READ_SEP] + [read_char(c) for c in gap] + [READ_END]
    output_start = len(initial_tape)
    program = _Program()
    program.move_heads_to(1, output_start)
    for c in s:
        program.step(
            head0_move=MOVE_RIGHT,
            head1_move=MOVE_RIGHT,
            head1_write=write_char(c),
        )
    program.finish()
    return _simulate("recall", initial_tape, program, expected_output=s)


def generate_reverse(s):
    initial_tape = _initial_tape(READ_REVERSE, s)
    scratch_start = len(initial_tape)
    program = _Program()

    # Phase 1: copy the original source to scratch after [END].
    program.move_heads_to(1, scratch_start)
    for c in s:
        program.step(
            head0_move=MOVE_RIGHT,
            head1_move=MOVE_RIGHT,
            head1_write=write_char(c),
        )

    # Phase 2: read scratch from right to left and overwrite source in place from
    # left to right. This leaves the scratch copy intact after [END].
    program.move_heads_to(1, scratch_start + len(s) - 1)
    for c in reversed(s):
        program.step(
            head0_move=MOVE_RIGHT,
            head1_move=MOVE_LEFT,
            head0_write=write_char(c),
        )
    program.finish()
    return _simulate("reverse", initial_tape, program, expected_output=s[::-1], output_start=1)


BRANCH_TASKS = {
    "copy": generate_copy,
    "mirror": generate_mirror,
    "recall": generate_recall,
    "reverse": generate_reverse,
}
DEFAULT_TASKS = ["copy", "mirror", "recall", "reverse"]


def _gen_strings(n_examples, min_len, max_len, seed):
    rng = random.Random(seed)
    return [
        "".join(rng.choice(SYMBOLS) for _ in range(rng.randint(min_len, max_len)))
        for _ in range(n_examples)
    ]


TRAIN_STRINGS = _gen_strings(100000, 4, 32, seed=42)
VAL_STRINGS = _gen_strings(200, 4, 32, seed=123)
TEST_LONG_STRINGS = _gen_strings(200, 48, 64, seed=456)


class TickTaskDataset:
    def __init__(self, strings, tasks=DEFAULT_TASKS):
        self.items = []
        for s in strings:
            for task in tasks:
                if task not in BRANCH_TASKS:
                    raise ValueError(f"unknown task {task!r}")
                self.items.append((task, s))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        task, s = self.items[idx]
        return BRANCH_TASKS[task](s)


class TickTaskDataLoader:
    def __init__(self, dataset, batch_size, pad_to_multiple=32, shuffle=True, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pad_to_multiple = pad_to_multiple
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
        import torch

        seq_len = max(len(t.target_head0_moves) for t in batch)
        padded_len = ((seq_len + self.pad_to_multiple - 1) // self.pad_to_multiple) * self.pad_to_multiple

        head0_reads = []
        head1_reads = []
        prev_head0_moves = []
        prev_head1_moves = []
        prev_head0_writes = []
        prev_head1_writes = []
        target_head0_moves = []
        target_head1_moves = []
        target_head0_writes = []
        target_head1_writes = []
        loss_mask = []

        for traj in batch:
            n = len(traj.target_head0_moves)
            pad_len = padded_len - n
            head0_reads.append(traj.head0_reads + [READ_PAD] * pad_len)
            head1_reads.append(traj.head1_reads + [READ_PAD] * pad_len)
            prev_head0_moves.append(traj.prev_head0_moves + [MOVE_PAD] * pad_len)
            prev_head1_moves.append(traj.prev_head1_moves + [MOVE_PAD] * pad_len)
            prev_head0_writes.append(traj.prev_head0_writes + [WRITE_PAD] * pad_len)
            prev_head1_writes.append(traj.prev_head1_writes + [WRITE_PAD] * pad_len)
            target_head0_moves.append(traj.target_head0_moves + [MOVE_PAD] * pad_len)
            target_head1_moves.append(traj.target_head1_moves + [MOVE_PAD] * pad_len)
            target_head0_writes.append(traj.target_head0_writes + [WRITE_PAD] * pad_len)
            target_head1_writes.append(traj.target_head1_writes + [WRITE_PAD] * pad_len)
            loss_mask.append([1] * n + [0] * pad_len)

        while len(head0_reads) < self.batch_size:
            head0_reads.append([READ_PAD] * padded_len)
            head1_reads.append([READ_PAD] * padded_len)
            prev_head0_moves.append([MOVE_PAD] * padded_len)
            prev_head1_moves.append([MOVE_PAD] * padded_len)
            prev_head0_writes.append([WRITE_PAD] * padded_len)
            prev_head1_writes.append([WRITE_PAD] * padded_len)
            target_head0_moves.append([MOVE_PAD] * padded_len)
            target_head1_moves.append([MOVE_PAD] * padded_len)
            target_head0_writes.append([WRITE_PAD] * padded_len)
            target_head1_writes.append([WRITE_PAD] * padded_len)
            loss_mask.append([0] * padded_len)

        return (
            torch.tensor(head0_reads, dtype=torch.long),
            torch.tensor(head1_reads, dtype=torch.long),
            torch.tensor(prev_head0_moves, dtype=torch.long),
            torch.tensor(prev_head1_moves, dtype=torch.long),
            torch.tensor(prev_head0_writes, dtype=torch.long),
            torch.tensor(prev_head1_writes, dtype=torch.long),
            torch.tensor(target_head0_moves, dtype=torch.long),
            torch.tensor(target_head1_moves, dtype=torch.long),
            torch.tensor(target_head0_writes, dtype=torch.long),
            torch.tensor(target_head1_writes, dtype=torch.long),
            torch.tensor(loss_mask, dtype=torch.float32),
        )

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


if __name__ == "__main__":
    for traj in [generate_copy("abc"), generate_mirror("abc"), generate_recall("abc"), generate_reverse("abc")]:
        print(traj.task, traj.final_output, len(traj.target_head0_moves))
        print("h0 reads", [READ_TOKEN_NAMES[x] for x in traj.head0_reads])
        print("h0 moves", [MOVE_TOKEN_NAMES[x] for x in traj.target_head0_moves])
        print("h0 writes", [WRITE_TOKEN_NAMES[x] for x in traj.target_head0_writes])
        print("h1 reads", [READ_TOKEN_NAMES[x] for x in traj.head1_reads])
        print("h1 moves", [MOVE_TOKEN_NAMES[x] for x in traj.target_head1_moves])
        print("h1 writes", [WRITE_TOKEN_NAMES[x] for x in traj.target_head1_writes])
