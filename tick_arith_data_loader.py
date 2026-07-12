"""Two-head, one-tape arithmetic tasks.

Each row is one machine tick. Both heads read the shared tape, then may write
their current cells and move independently. Arithmetic is performed in scratch
cells at negative positions; the finished answer is copied after [END].

Initial tape layout:
  [ADD|MUL] digits(a) [SEP] digits(b) [END] blanks...
"""

from dataclasses import dataclass
import random


# --- Tape/read vocabulary ----------------------------------------------------
READ_PAD = 0
READ_BLANK = 1
READ_ADD = 2
READ_MUL = 3
READ_SEP = 4
READ_END = 5
READ_USED = 6
READ_DIGIT_BASE = 7
DIGITS = list("0123456789")
READ_VOCAB_SIZE = READ_DIGIT_BASE + len(DIGITS)


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
WRITE_ADD = 4
WRITE_MUL = 5
WRITE_SEP = 6
WRITE_END = 7
WRITE_USED = 8
WRITE_DIGIT_BASE = 9
WRITE_VOCAB_SIZE = WRITE_DIGIT_BASE + len(DIGITS)


READ_TOKEN_NAMES = {
    READ_PAD: "[PAD]",
    READ_BLANK: "[BLANK]",
    READ_ADD: "[ADD]",
    READ_MUL: "[MUL]",
    READ_SEP: "[SEP]",
    READ_END: "[END]",
    READ_USED: "[USED]",
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
    WRITE_ADD: "[WRITE_ADD]",
    WRITE_MUL: "[WRITE_MUL]",
    WRITE_SEP: "[WRITE_SEP]",
    WRITE_END: "[WRITE_END]",
    WRITE_USED: "[WRITE_USED]",
}
for i, digit in enumerate(DIGITS):
    READ_TOKEN_NAMES[READ_DIGIT_BASE + i] = f"[{digit}]"
    WRITE_TOKEN_NAMES[WRITE_DIGIT_BASE + i] = f"[WRITE_{digit}]"


def read_digit(digit):
    return READ_DIGIT_BASE + int(digit)


def write_digit(digit):
    return WRITE_DIGIT_BASE + int(digit)


def write_to_read(write):
    if write == WRITE_NOOP:
        return None
    if write == WRITE_BLANK:
        return READ_BLANK
    if write == WRITE_ADD:
        return READ_ADD
    if write == WRITE_MUL:
        return READ_MUL
    if write == WRITE_SEP:
        return READ_SEP
    if write == WRITE_END:
        return READ_END
    if write == WRITE_USED:
        return READ_USED
    if WRITE_DIGIT_BASE <= write < WRITE_VOCAB_SIZE:
        return READ_DIGIT_BASE + (write - WRITE_DIGIT_BASE)
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
    answer: str
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
    if token is not None:
        tape[pos] = token


def _validate_number(s):
    if not s or any(digit not in DIGITS for digit in s):
        raise ValueError(f"invalid decimal number: {s!r}")
    if len(s) > 1 and s.startswith("0"):
        raise ValueError(f"leading zeros are not allowed: {s!r}")
    return s


def _gen_number(rng, min_len, max_len):
    length = rng.randint(min_len, max_len)
    if length == 1:
        return rng.choice(DIGITS)
    return rng.choice("123456789") + "".join(
        rng.choice(DIGITS) for _ in range(length - 1)
    )


def _gen_items(n_examples, min_len, max_len, seed):
    rng = random.Random(seed)
    return [
        (_gen_number(rng, min_len, max_len), _gen_number(rng, min_len, max_len))
        for _ in range(n_examples)
    ]


def _initial_tape(op_token, a, b):
    return (
        [op_token]
        + [read_digit(digit) for digit in a]
        + [READ_SEP]
        + [read_digit(digit) for digit in b]
        + [READ_END]
    )


def _simulate(task, initial_tape, program, answer):
    tape = {pos: token for pos, token in enumerate(initial_tape)}
    head0 = 0
    head1 = 0
    head0_reads = []
    head1_reads = []

    for i in range(len(program.head0_moves)):
        head0_reads.append(_read_tape(tape, head0))
        head1_reads.append(_read_tape(tape, head1))

        head0_write = program.head0_writes[i]
        head1_write = program.head1_writes[i]
        if head0 == head1 and head0_write != WRITE_NOOP and head1_write != WRITE_NOOP:
            if head0_write != head1_write:
                raise AssertionError(f"{task}: conflicting writes at tape position {head0}")
        _apply_write(tape, head0, head0_write)
        _apply_write(tape, head1, head1_write)

        head0 += _move_delta(program.head0_moves[i])
        head1 += _move_delta(program.head1_moves[i])

    output_start = len(initial_tape)
    final_output = []
    for offset, expected_digit in enumerate(answer):
        token = _read_tape(tape, output_start + offset)
        expected = read_digit(expected_digit)
        if token != expected:
            raise AssertionError(
                f"{task}: output[{offset}] expected {READ_TOKEN_NAMES[expected]}, "
                f"got {READ_TOKEN_NAMES[token]}"
            )
        final_output.append(expected_digit)

    return TickTrajectory(
        task=task,
        head0_reads=head0_reads,
        head1_reads=head1_reads,
        target_head0_moves=list(program.head0_moves),
        target_head1_moves=list(program.head1_moves),
        target_head0_writes=list(program.head0_writes),
        target_head1_writes=list(program.head1_writes),
        answer=answer,
        final_output="".join(final_output),
    )


def _copy_paired_scratch_to_output_with_head1(
    program,
    scratch_start,
    output_start,
    answer,
):
    program.move_heads_to(output_start, scratch_start)
    for digit in answer:
        program.step(
            head0_move=MOVE_RIGHT,
            head1_move=MOVE_RIGHT,
            head0_write=write_digit(digit),
        )
        # Skip the carry cell. After the last digit this walk continues onto
        # [ADD], whose read is the tape-visible halt signal.
        program.step(head1_move=MOVE_RIGHT)


def _run_move_scripts(program, head0_moves, head1_moves, head1_writes=None):
    """Advance both heads along independent move scripts in parallel;
    whichever head finishes first waits in place until the other is done."""
    for i in range(max(len(head0_moves), len(head1_moves))):
        program.step(
            head0_move=head0_moves[i] if i < len(head0_moves) else MOVE_STAY,
            head1_move=head1_moves[i] if i < len(head1_moves) else MOVE_STAY,
            head1_write=(
                head1_writes[i]
                if head1_writes is not None and i < len(head1_writes)
                else WRITE_NOOP
            ),
        )


def generate_add(a, b):
    """Add with one explicit carry cell after every scratch digit."""
    a = _validate_number(str(a))
    b = _validate_number(str(b))
    answer = str(int(a) + int(b))
    initial_tape = _initial_tape(READ_ADD, a, b)
    output_start = len(initial_tape)
    sep_pos = len(a) + 1
    b_end = sep_pos + len(b)
    end_pos = b_end + 1
    width = max(len(a), len(b))
    program = _Program()

    def main_pos(column):
        # Column 0 is the LSD. Each digit's outgoing carry is immediately right.
        return -2 - 2 * column

    def carry_pos(column):
        return main_pos(column) + 1

    # Head 0 discovers [SEP]. Head 1 reserves the LSD carry cell at -1,
    # then both heads move left onto a's LSD and its scratch main cell.
    program.move_heads_to(sep_pos, 0)
    program.step(head1_move=MOVE_LEFT)
    program.step(head0_move=MOVE_LEFT, head1_move=MOVE_LEFT)
    for digit in reversed(a):
        # Same-tick echo into the main cell, leaving the carry cell blank.
        # Head 0 slides one digit left per pair; reading [ADD] ends the copy.
        program.step(
            head0_move=MOVE_LEFT,
            head1_move=MOVE_LEFT,
            head1_write=write_digit(digit),
        )
        program.step(head1_move=MOVE_LEFT)

    # Head 0 discovers b's right boundary by reading [END], then steps left
    # onto b's LSD. Head 1 walks right until it reads [ADD] and backs up two
    # cells onto the copied a's LSD.
    _run_move_scripts(
        program,
        [MOVE_RIGHT] * end_pos,
        [MOVE_RIGHT] * -program.head1 + [MOVE_LEFT] * 2,
    )
    program.step(head0_move=MOVE_LEFT)

    carry = 0
    for column in range(width + 1):
        a_index = len(a) - 1 - column
        b_index = len(b) - 1 - column
        copied_a_pos = main_pos(column)
        b_pos = sep_pos + 1 + b_index if b_index >= 0 else sep_pos
        a_digit = int(a[a_index]) if a_index >= 0 else 0
        b_digit = int(b[b_index]) if b_index >= 0 else 0

        if program.head0 != b_pos or program.head1 != copied_a_pos:
            raise AssertionError("addition heads are not aligned with the next column")

        # The incoming carry was read from the less-significant digit's carry
        # cell while moving here. Update this main digit, then move right and
        # materialize this digit's outgoing carry in its own paired cell.
        total = a_digit + b_digit + carry
        result_digit = total % 10
        carry = total // 10
        program.step(
            head1_move=MOVE_RIGHT,
            head1_write=write_digit(result_digit),
        )
        if program.head1 != carry_pos(column):
            raise AssertionError("addition head missed the outgoing carry cell")
        program.step(head1_write=write_digit(carry))

        if column < width:
            next_b_index = len(b) - 2 - column
            next_b_pos = (
                sep_pos + 1 + next_b_index if next_b_index >= 0 else sep_pos
            )

            # Read back the explicit outgoing carry, then cross the current
            # main cell and the next pair's carry cell to reach its main digit.
            # Head 0 retires the consumed b digit as it leaves, so the later
            # walk to the output crosses [USED] cells instead of fresh digits.
            program.step(
                head0_move=_move_toward(program.head0, next_b_pos),
                head0_write=WRITE_USED if b_index >= 0 else WRITE_NOOP,
                head1_move=MOVE_LEFT,
            )
            program.step(head1_move=MOVE_LEFT)
            program.step(head1_move=MOVE_LEFT)

    scratch_start = main_pos(width) if len(answer) > width else main_pos(width - 1)
    _copy_paired_scratch_to_output_with_head1(
        program,
        scratch_start,
        output_start,
        answer,
    )
    program.finish()
    return _simulate("add", initial_tape, program, answer)


def generate_mul(a, b):
    """Long multiplication with explicit ``[MAIN, CARRY, WIP]`` blocks."""
    a = _validate_number(str(a))
    b = _validate_number(str(b))
    answer = str(int(a) * int(b))
    initial_tape = _initial_tape(READ_MUL, a, b)
    output_start = len(initial_tape)
    sep_pos = len(a) + 1
    b_start = sep_pos + 1
    full_width = len(a) + len(b)
    accumulator = [0] * full_width
    program = _Program()

    def main_pos(index):
        # Index 0 is the LSD. Each block is [MAIN, CARRY, WIP] left-to-right.
        return -3 - 3 * index

    def carry_pos(index):
        return main_pos(index) + 1

    def wip_pos(index):
        return main_pos(index) + 2

    for b_index in range(len(b) - 1, -1, -1):
        b_pos = b_start + b_index
        b_digit = int(b[b_index])
        shift = len(b) - 1 - b_index

        # Head 0 discovers the next b digit by walking to [USED]/[END] and
        # stepping left. Head 1 uses the previous row's [USED] WIP landmark to
        # locate this row's shifted starting block.
        head0_moves = [MOVE_RIGHT] * (b_pos + 1 - program.head0) + [MOVE_LEFT]
        head1_writes = None
        if shift == 0:
            head1_moves = [MOVE_LEFT]
        else:
            n_right = wip_pos(shift - 1) - program.head1
            head1_moves = [MOVE_RIGHT] * n_right + [MOVE_LEFT] * 3
            head1_writes = [WRITE_NOOP] * n_right + [WRITE_BLANK]
        _run_move_scripts(program, head0_moves, head1_moves, head1_writes)
        if program.head0 != b_pos or program.head1 != wip_pos(shift):
            raise AssertionError("multiplication fetch missed b or the WIP cell")

        # Same-tick echo: retire b_j on the input side and seed this row's WIP.
        program.step(head0_write=WRITE_USED, head1_write=write_digit(b_digit))

        # Head 1 keeps rereading b_j on the WIP cell until head 0 reads [SEP]
        # on its last transit tick; only then does it mark the starting WIP as
        # the row landmark, materialize the zero carry-in, and reach MAIN, two
        # ticks after head 0 parks on a's LSD. This keeps b_j at most three
        # ticks old at the first MAC regardless of how far head 0 travels.
        n_wait = b_pos - len(a) - 1
        _run_move_scripts(
            program,
            [MOVE_LEFT] * (b_pos - len(a)),
            [MOVE_STAY] * n_wait + [MOVE_LEFT, MOVE_STAY, MOVE_LEFT],
            [WRITE_NOOP] * n_wait + [WRITE_USED, write_digit(0), WRITE_NOOP],
        )
        if program.head0 != len(a) or program.head1 != main_pos(shift):
            raise AssertionError("multiplication row did not align on its LSD")

        carry = 0
        for column in range(len(a)):
            a_index = len(a) - 1 - column
            a_digit = int(a[a_index])
            acc_index = shift + column
            if program.head0 != 1 + a_index or program.head1 != main_pos(acc_index):
                raise AssertionError("multiplication heads are not aligned")

            total = accumulator[acc_index] + a_digit * b_digit + carry
            accumulator[acc_index] = total % 10
            carry = total // 10

            # Update MAIN and move right into its paired outgoing-carry cell.
            program.step(
                head0_move=MOVE_LEFT,
                head1_move=MOVE_RIGHT,
                head1_write=write_digit(accumulator[acc_index]),
            )
            if program.head1 != carry_pos(acc_index):
                raise AssertionError("multiplication missed the carry cell")

            # Materialize and reread carry before walking into the next block.
            program.step(head1_write=write_digit(carry))
            program.step(head1_move=MOVE_LEFT)
            program.step(head1_move=MOVE_LEFT)

            next_index = acc_index + 1
            if program.head1 != wip_pos(next_index):
                raise AssertionError("multiplication missed the next WIP cell")

            # Materialize and reread b_j, then copy the outgoing carry into
            # the next block and reread it immediately before reaching MAIN.
            program.step(head1_write=write_digit(b_digit))
            program.step(
                head1_move=MOVE_LEFT,
                head1_write=WRITE_BLANK,
            )
            program.step(head1_write=write_digit(carry))
            program.step(head1_move=MOVE_LEFT)

        # One extra zero-input block turns the final explicit carry into a
        # normal persistent MAIN digit. This target is untouched by prior rows.
        final_index = shift + len(a)
        if accumulator[final_index] != 0:
            raise AssertionError("multiplication carry target is not empty")
        accumulator[final_index] = carry
        if program.head0 != 0 or program.head1 != main_pos(final_index):
            raise AssertionError("multiplication epilogue is not aligned")
        program.step(
            head1_move=MOVE_RIGHT,
            head1_write=write_digit(carry),
        )
        program.step(head1_write=write_digit(0))

    scratch_answer = "".join(str(digit) for digit in reversed(accumulator))
    scratch_answer = scratch_answer.lstrip("0") or "0"
    if scratch_answer != answer:
        raise AssertionError(
            f"multiplication scratch expected {answer}, got {scratch_answer}"
        )

    # Termination fetch: all b digits are [USED], so stepping left from the
    # first remaining boundary lands head 0 on [SEP].
    head0_moves = [MOVE_RIGHT] * (b_start - program.head0) + [MOVE_LEFT]
    n_right = wip_pos(len(b) - 1) - program.head1
    head1_moves = [MOVE_RIGHT] * n_right + [MOVE_LEFT] * 3
    head1_writes = [WRITE_NOOP] * n_right + [WRITE_BLANK]
    _run_move_scripts(program, head0_moves, head1_moves, head1_writes)
    if program.head0 != sep_pos or program.head1 != wip_pos(len(b)):
        raise AssertionError("termination walks missed [SEP] or the landmark")

    # Head 0 walks to the output region. Head 1 discovers the left scratch
    # boundary, returns to the most-significant MAIN, then skips leading zeros.
    msd_pos = main_pos(full_width - 1)
    head0_moves = [MOVE_RIGHT] * (output_start - sep_pos)
    head1_moves = [MOVE_LEFT] * (program.head1 - (msd_pos - 2)) + [MOVE_RIGHT] * 2
    head1_moves += [MOVE_RIGHT, MOVE_RIGHT, MOVE_RIGHT] * (
        full_width - len(answer)
    )
    if answer == "0":
        # A lone zero MAIN looks exactly like another leading zero to skip;
        # overshoot onto [MUL] and back up so the stop is tape-triggered.
        head1_moves += [MOVE_RIGHT] * 3 + [MOVE_LEFT] * 3
    _run_move_scripts(program, head0_moves, head1_moves)
    if program.head0 != output_start or program.head1 != main_pos(len(answer) - 1):
        raise AssertionError("copy walks missed the output start or the MSD")

    # Copy MAIN cells only, skipping each CARRY and WIP cell. After the last
    # digit the skip walk lands on [MUL], the tape-visible halt signal.
    for digit in answer:
        program.step(
            head0_move=MOVE_RIGHT,
            head1_move=MOVE_RIGHT,
            head0_write=write_digit(digit),
        )
        program.step(head1_move=MOVE_RIGHT)
        program.step(head1_move=MOVE_RIGHT)

    program.finish()
    return _simulate("mul", initial_tape, program, answer)


TASK_GENERATORS = {
    "add": generate_add,
    "mul": generate_mul,
}
DEFAULT_TASKS = ["add", "mul"]


TRAIN_ADD = _gen_items(100000, 1, 8, seed=42)
VAL_ADD = _gen_items(500, 1, 8, seed=123)
TEST_LONG_ADD = _gen_items(500, 12, 16, seed=456)

TRAIN_MUL = _gen_items(100000, 1, 4, seed=43)
VAL_MUL = _gen_items(500, 1, 4, seed=124)
TEST_LONG_MUL = _gen_items(500, 6, 8, seed=457)


class ArithmeticTickDataset:
    def __init__(
        self,
        add_items=TRAIN_ADD,
        mul_items=TRAIN_MUL,
        tasks=DEFAULT_TASKS,
        shuffle_seed=None,
    ):
        self.items = []
        for task in tasks:
            if task not in TASK_GENERATORS:
                raise ValueError(f"unknown task {task!r}")
            items = add_items if task == "add" else mul_items
            self.items.extend((task, a, b) for a, b in items)
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        task, a, b = self.items[idx]
        return TASK_GENERATORS[task](a, b)


# --- Curriculum ---------------------------------------------------------------
# Stages run in order: add starts at single digits and grows to full length,
# then mul does the same, and the final stage mixes both ops at the full
# training ranges. Counts are per-task example counts.
CURRICULUM_STAGES = [
    (["add"], (1, 1), None, 5000),
    (["add"], (1, 2), None, 10000),
    (["add"], (1, 4), None, 15000),
    (["add"], (1, 8), None, 20000),
    (["mul"], None, (1, 1), 5000),
    (["mul"], None, (1, 2), 15000),
    (["mul"], None, (1, 4), 25000),
    (["add", "mul"], (1, 8), (1, 4), 100000),
]


def curriculum_dataset(stages=CURRICULUM_STAGES, seed=2000):
    """One dataset whose items follow the curriculum order: each stage is
    shuffled internally, stages are concatenated in order."""
    dataset = ArithmeticTickDataset(add_items=[], mul_items=[])
    for i, (tasks, add_len, mul_len, n_examples) in enumerate(stages):
        base = seed + 3 * i
        add_items = (
            _gen_items(n_examples, *add_len, seed=base) if "add" in tasks else []
        )
        mul_items = (
            _gen_items(n_examples, *mul_len, seed=base + 1) if "mul" in tasks else []
        )
        stage = ArithmeticTickDataset(add_items, mul_items, tasks, shuffle_seed=base + 2)
        dataset.items.extend(stage.items)
    return dataset


class ArithmeticTickDataLoader:
    def __init__(self, dataset, batch_size, pad_to_multiple=32):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pad_to_multiple = pad_to_multiple

    def __iter__(self):
        for start in range(0, len(self.dataset), self.batch_size):
            end = min(start + self.batch_size, len(self.dataset))
            batch = [self.dataset[i] for i in range(start, end)]
            yield self._collate(batch)

    def _collate(self, batch):
        import torch

        seq_len = max(len(t.target_head0_moves) for t in batch)
        padded_len = (
            (seq_len + self.pad_to_multiple - 1) // self.pad_to_multiple
        ) * self.pad_to_multiple

        fields = {
            "head0_reads": (READ_PAD, []),
            "head1_reads": (READ_PAD, []),
            "prev_head0_moves": (MOVE_PAD, []),
            "prev_head1_moves": (MOVE_PAD, []),
            "prev_head0_writes": (WRITE_PAD, []),
            "prev_head1_writes": (WRITE_PAD, []),
            "target_head0_moves": (MOVE_PAD, []),
            "target_head1_moves": (MOVE_PAD, []),
            "target_head0_writes": (WRITE_PAD, []),
            "target_head1_writes": (WRITE_PAD, []),
        }
        loss_mask = []

        for trajectory in batch:
            n = len(trajectory.target_head0_moves)
            pad_len = padded_len - n
            for name, (pad_token, rows) in fields.items():
                rows.append(getattr(trajectory, name) + [pad_token] * pad_len)
            loss_mask.append([1] * n + [0] * pad_len)

        while len(loss_mask) < self.batch_size:
            for pad_token, rows in fields.values():
                rows.append([pad_token] * padded_len)
            loss_mask.append([0] * padded_len)

        tensors = [
            torch.tensor(rows, dtype=torch.long) for _, rows in fields.values()
        ]
        tensors.append(torch.tensor(loss_mask, dtype=torch.float32))
        return tuple(tensors)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def _print_trajectory(label, trajectory):
    print(
        f"{label} = {trajectory.final_output} "
        f"({len(trajectory.target_head0_moves)} ticks)"
    )
    print(
        f"{'tick':>4}  "
        f"{'head0 read':>12} {'head0 write':>14} {'head0 move':>10}  "
        f"{'head1 read':>12} {'head1 write':>14} {'head1 move':>10}"
    )
    for tick, values in enumerate(zip(
        trajectory.head0_reads,
        trajectory.target_head0_writes,
        trajectory.target_head0_moves,
        trajectory.head1_reads,
        trajectory.target_head1_writes,
        trajectory.target_head1_moves,
    )):
        head0_read, head0_write, head0_move, head1_read, head1_write, head1_move = values
        print(
            f"{tick:4d}  "
            f"{READ_TOKEN_NAMES[head0_read]:>12} "
            f"{WRITE_TOKEN_NAMES[head0_write]:>14} "
            f"{MOVE_TOKEN_NAMES[head0_move]:>10}  "
            f"{READ_TOKEN_NAMES[head1_read]:>12} "
            f"{WRITE_TOKEN_NAMES[head1_write]:>14} "
            f"{MOVE_TOKEN_NAMES[head1_move]:>10}"
        )
    print()


if __name__ == "__main__":
    for a, b in [("123", "456"), ("999", "1"), ("12", "9876")]:
        _print_trajectory(f"{a} + {b}", generate_add(a, b))
    for a, b in [("12", "34"), ("999", "0"), ("1234", "5678")]:
        _print_trajectory(f"{a} * {b}", generate_mul(a, b))
