"""Tick-level Turing-machine tasks with implicit tape reads.

Each training row is one machine tick. There is no READ action. On every tick the
model receives:
  - current input-tape symbol
  - current output-tape symbol
  - previous action

The supervised target is the next action to execute. The deterministic simulator
applies that action to update tape heads and output-tape contents before the next
tick is recorded.
"""

from dataclasses import dataclass
import random
import torch


# --- Read vocabulary: what the two tape heads observe -------------------------
READ_PAD = 0
READ_BLANK = 1
READ_COPY = 2
READ_MIRROR = 3
READ_REPEAT = 4

SYMBOLS = list("abcdefgh")
READ_CHAR_BASE = 5
READ_VOCAB_SIZE = READ_CHAR_BASE + len(SYMBOLS)


# --- Action vocabulary: what the model chooses -------------------------------
ACTION_PAD = 0
ACTION_START = 1
ACTION_DONE = 2
ACTION_MOVE_INPUT_LEFT = 3
ACTION_MOVE_INPUT_RIGHT = 4
ACTION_MOVE_OUTPUT_LEFT = 5
ACTION_MOVE_OUTPUT_RIGHT = 6
ACTION_WRITE_BASE = 7
ACTION_VOCAB_SIZE = ACTION_WRITE_BASE + len(SYMBOLS)


READ_TOKEN_NAMES = {
    READ_PAD: "[PAD]",
    READ_BLANK: "[BLANK]",
    READ_COPY: "[COPY]",
    READ_MIRROR: "[MIRROR]",
    READ_REPEAT: "[REPEAT]",
}
ACTION_TOKEN_NAMES = {
    ACTION_PAD: "[PAD]",
    ACTION_START: "[START]",
    ACTION_DONE: "[DONE]",
    ACTION_MOVE_INPUT_LEFT: "[MOVE_INPUT_LEFT]",
    ACTION_MOVE_INPUT_RIGHT: "[MOVE_INPUT_RIGHT]",
    ACTION_MOVE_OUTPUT_LEFT: "[MOVE_OUTPUT_LEFT]",
    ACTION_MOVE_OUTPUT_RIGHT: "[MOVE_OUTPUT_RIGHT]",
}
for i, c in enumerate(SYMBOLS):
    READ_TOKEN_NAMES[READ_CHAR_BASE + i] = f"[{c}]"
    ACTION_TOKEN_NAMES[ACTION_WRITE_BASE + i] = f"[WRITE_{c}]"


def read_char(sym):
    return READ_CHAR_BASE + SYMBOLS.index(sym)


def write_action(sym):
    return ACTION_WRITE_BASE + SYMBOLS.index(sym)


def write_to_read(action):
    return READ_CHAR_BASE + (action - ACTION_WRITE_BASE)


@dataclass
class TickTrajectory:
    task: str
    input_reads: list[int]
    output_reads: list[int]
    target_actions: list[int]
    @property
    def prev_actions(self):
        return [ACTION_START] + self.target_actions[:-1]


def _read_input_tape(input_tape, pos):
    if 0 <= pos < len(input_tape):
        return input_tape[pos]
    return READ_BLANK


def _simulate(task, input_tape, target_actions):
    input_head = 0
    output_head = 0
    output_tape = {}
    input_reads = []
    output_reads = []
    for i, action in enumerate(target_actions):
        input_reads.append(_read_input_tape(input_tape, input_head))
        output_reads.append(output_tape.get(output_head, READ_BLANK))
        if action == ACTION_DONE:
            pass
        elif action == ACTION_MOVE_INPUT_LEFT:
            input_head -= 1
        elif action == ACTION_MOVE_INPUT_RIGHT:
            input_head += 1
        elif action == ACTION_MOVE_OUTPUT_LEFT:
            output_head -= 1
        elif action == ACTION_MOVE_OUTPUT_RIGHT:
            output_head += 1
        elif ACTION_WRITE_BASE <= action < ACTION_VOCAB_SIZE:
            output_tape[output_head] = write_to_read(action)
        else:
            raise ValueError(f"invalid action {action}")

    return TickTrajectory(
        task=task,
        input_reads=input_reads,
        output_reads=output_reads,
        target_actions=list(target_actions),
    )


def generate_copy(s):
    input_tape = [READ_COPY] + [read_char(c) for c in s]
    actions = [ACTION_MOVE_INPUT_RIGHT]
    for c in s:
        actions += [write_action(c), ACTION_MOVE_INPUT_RIGHT, ACTION_MOVE_OUTPUT_RIGHT]
    actions += [ACTION_DONE]
    return _simulate("copy", input_tape, actions)


def generate_mirror(s):
    input_tape = [READ_MIRROR] + [read_char(c) for c in s]
    actions = [ACTION_MOVE_INPUT_RIGHT]
    for c in s:
        actions += [write_action(c), ACTION_MOVE_INPUT_RIGHT, ACTION_MOVE_OUTPUT_RIGHT]
    actions += [ACTION_MOVE_INPUT_LEFT]
    for c in reversed(s):
        actions += [write_action(c), ACTION_MOVE_INPUT_LEFT, ACTION_MOVE_OUTPUT_RIGHT]
    actions += [ACTION_DONE]
    return _simulate("mirror", input_tape, actions)


def generate_repeat(s):
    input_tape = [READ_REPEAT] + [read_char(c) for c in s]
    actions = [ACTION_MOVE_INPUT_RIGHT]
    for c in s:
        actions += [write_action(c), ACTION_MOVE_INPUT_RIGHT, ACTION_MOVE_OUTPUT_RIGHT]
    for _ in s:
        actions += [ACTION_MOVE_INPUT_LEFT]
    for c in s:
        actions += [write_action(c), ACTION_MOVE_INPUT_RIGHT, ACTION_MOVE_OUTPUT_RIGHT]
    actions += [ACTION_DONE]
    return _simulate("repeat", input_tape, actions)



BRANCH_TASKS = {
    "copy": generate_copy,
    "mirror": generate_mirror,
    "repeat": generate_repeat,
}
DEFAULT_TASKS = ["copy", "mirror", "repeat"]


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
        self.tasks = tuple(tasks)
        data = []
        branch = [t for t in self.tasks if t in BRANCH_TASKS]
        for s in strings:
            for t in branch:
                data.append(BRANCH_TASKS[t](s))
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


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
        seq_len = max(len(t.target_actions) for t in batch)
        padded_len = ((seq_len + self.pad_to_multiple - 1) // self.pad_to_multiple) * self.pad_to_multiple

        input_reads = []
        output_reads = []
        prev_actions = []
        target_actions = []
        loss_mask = []
        for traj in batch:
            n = len(traj.target_actions)
            pad_len = padded_len - n
            input_reads.append(traj.input_reads + [READ_PAD] * pad_len)
            output_reads.append(traj.output_reads + [READ_PAD] * pad_len)
            prev_actions.append(traj.prev_actions + [ACTION_PAD] * pad_len)
            target_actions.append(traj.target_actions + [ACTION_PAD] * pad_len)
            loss_mask.append([1] * n + [0] * pad_len)

        while len(input_reads) < self.batch_size:
            input_reads.append([READ_PAD] * padded_len)
            output_reads.append([READ_PAD] * padded_len)
            prev_actions.append([ACTION_PAD] * padded_len)
            target_actions.append([ACTION_PAD] * padded_len)
            loss_mask.append([0] * padded_len)

        return (
            torch.tensor(input_reads, dtype=torch.long),
            torch.tensor(output_reads, dtype=torch.long),
            torch.tensor(prev_actions, dtype=torch.long),
            torch.tensor(target_actions, dtype=torch.long),
            torch.tensor(loss_mask, dtype=torch.float32),
        )

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


if __name__ == "__main__":
    for traj in [generate_copy("abc"), generate_mirror("abc"), generate_repeat("abc")]:
        print(traj.task, len(traj.target_actions), traj.target_actions[:12])
