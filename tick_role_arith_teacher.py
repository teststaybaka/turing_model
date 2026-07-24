"""Sequential two-channel arithmetic environment and corrective teacher."""

from dataclasses import dataclass
from itertools import product

from tick_role_arith_data_loader import (
    MOVE_LEFT,
    MOVE_RIGHT,
    MOVE_START,
    MOVE_STAY,
    MOVE_VOCAB_SIZE,
    READ_BLANK,
    READ_DIGIT_BASE,
    WRITE_NOOP,
    WRITE_START,
    WRITE_VOCAB_SIZE,
    ROLE_INPUT,
    ROLE_NONE,
    ROLE_RES,
    ROLE_USED,
    ROLE_WRITE_NOOP,
    ROLE_WRITE_START,
    ROLE_WRITE_VOCAB_SIZE,
    initial_role_tape,
    initial_value_tape,
    read_digit,
    role_write_to_read,
    write_to_read,
)


HEAD0_MOVE = 0
HEAD1_MOVE = 1
HEAD0_VALUE_WRITE = 2
HEAD1_VALUE_WRITE = 3
HEAD0_ROLE_WRITE = 4
HEAD1_ROLE_WRITE = 5
NUM_ACTION_FACTORS = 6

START_ACTION = (
    MOVE_START,
    MOVE_START,
    WRITE_START,
    WRITE_START,
    ROLE_WRITE_START,
    ROLE_WRITE_START,
)
NEUTRAL_ACTION = (
    MOVE_STAY,
    MOVE_STAY,
    WRITE_NOOP,
    WRITE_NOOP,
    ROLE_WRITE_NOOP,
    ROLE_WRITE_NOOP,
)


class InvalidAction(ValueError):
    pass


def _move_delta(move):
    if move == MOVE_LEFT:
        return -1
    if move == MOVE_RIGHT:
        return 1
    if move == MOVE_STAY:
        return 0
    raise InvalidAction(f"invalid move token {move}")


@dataclass
class TapeEvent:
    reads: tuple[int, int, int, int]
    action: tuple[int, ...]
    head_positions_before: tuple[int, int]
    head_positions_after: tuple[int, int]
    value_changes: tuple[tuple[int, int, int], ...]
    role_changes: tuple[tuple[int, int, int], ...]


class RoleTapeEnvironment:
    def __init__(self, task, a, b):
        if task not in ("add", "mul"):
            raise ValueError(f"unknown task {task!r}")
        self.task = task
        self.a = str(a)
        self.b = str(b)
        self.value_tape = initial_value_tape(task, self.a, self.b)
        self.role_tape = initial_role_tape(self.a, self.b)
        self.initial_values = dict(self.value_tape)
        self.initial_roles = dict(self.role_tape)
        self.output_start = len(self.a) + len(self.b) + 3
        self.head0 = 0
        self.head1 = 0
        self.prev_action = START_ACTION
        self.ticks = 0

    def clone(self):
        clone = object.__new__(RoleTapeEnvironment)
        clone.task = self.task
        clone.a = self.a
        clone.b = self.b
        clone.value_tape = dict(self.value_tape)
        clone.role_tape = dict(self.role_tape)
        clone.initial_values = self.initial_values
        clone.initial_roles = self.initial_roles
        clone.output_start = self.output_start
        clone.head0 = self.head0
        clone.head1 = self.head1
        clone.prev_action = self.prev_action
        clone.ticks = self.ticks
        return clone

    def reads(self):
        return (
            self.value_tape.get(self.head0, READ_BLANK),
            self.value_tape.get(self.head1, READ_BLANK),
            self.role_tape.get(self.head0, ROLE_NONE),
            self.role_tape.get(self.head1, ROLE_NONE),
        )

    def apply(self, action):
        action = tuple(int(token) for token in action)
        if len(action) != NUM_ACTION_FACTORS:
            raise InvalidAction(
                f"expected {NUM_ACTION_FACTORS} factors, got {len(action)}"
            )
        h0_move, h1_move, h0_value, h1_value, h0_role, h1_role = action
        if not (MOVE_STAY <= h0_move < MOVE_VOCAB_SIZE):
            raise InvalidAction(f"invalid head 0 move {h0_move}")
        if not (MOVE_STAY <= h1_move < MOVE_VOCAB_SIZE):
            raise InvalidAction(f"invalid head 1 move {h1_move}")
        if not (WRITE_NOOP <= h0_value < WRITE_VOCAB_SIZE):
            raise InvalidAction(f"invalid head 0 value write {h0_value}")
        if not (WRITE_NOOP <= h1_value < WRITE_VOCAB_SIZE):
            raise InvalidAction(f"invalid head 1 value write {h1_value}")
        if not (ROLE_WRITE_NOOP <= h0_role < ROLE_WRITE_VOCAB_SIZE):
            raise InvalidAction(f"invalid head 0 role write {h0_role}")
        if not (ROLE_WRITE_NOOP <= h1_role < ROLE_WRITE_VOCAB_SIZE):
            raise InvalidAction(f"invalid head 1 role write {h1_role}")

        if self.head0 == self.head1:
            if h0_value != WRITE_NOOP and h1_value != WRITE_NOOP:
                if h0_value != h1_value:
                    raise InvalidAction("conflicting value writes")
            if h0_role != ROLE_WRITE_NOOP and h1_role != ROLE_WRITE_NOOP:
                if h0_role != h1_role:
                    raise InvalidAction("conflicting role writes")

        reads = self.reads()
        before = (self.head0, self.head1)
        value_changes = []
        role_changes = []
        for position, value_write in (
            (self.head0, h0_value),
            (self.head1, h1_value),
        ):
            value = write_to_read(value_write)
            if value is not None:
                old_value = self.value_tape.get(position, READ_BLANK)
                if old_value != value:
                    value_changes.append((position, old_value, value))
                if value == READ_BLANK:
                    self.value_tape.pop(position, None)
                else:
                    self.value_tape[position] = value

        for position, role_write in (
            (self.head0, h0_role),
            (self.head1, h1_role),
        ):
            role = role_write_to_read(role_write)
            if role is not None:
                old_role = self.role_tape.get(position, ROLE_NONE)
                if old_role != role:
                    role_changes.append((position, old_role, role))
                if role == ROLE_NONE:
                    self.role_tape.pop(position, None)
                else:
                    self.role_tape[position] = role

        self.head0 += _move_delta(h0_move)
        self.head1 += _move_delta(h1_move)
        self.prev_action = action
        self.ticks += 1
        return TapeEvent(
            reads=reads,
            action=action,
            head_positions_before=before,
            head_positions_after=(self.head0, self.head1),
            value_changes=tuple(value_changes),
            role_changes=tuple(role_changes),
        )

@dataclass
class TeacherDecision:
    executed_action: tuple[int, ...]
    reward: float
    intervened: bool
    success: bool
    reason: str
    budget_cost: int


class RoleAwareTeacher:
    """Protect the input and reward valid scratch/output progress."""

    STEP_PENALTY = -0.01
    INVALID_PENALTY = -1.0
    CHECKPOINT_REWARD = 0.25
    OUTPUT_REWARD = 10.0

    def __init__(self, task, a, b):
        self.task = task
        self.a = str(a)
        self.b = str(b)
        self.answer = str(
            int(self.a) + int(self.b)
            if task == "add"
            else int(self.a) * int(self.b)
        )
        self.scratch_width = (
            max(len(self.a), len(self.b)) + 1
            if task == "add"
            else len(self.a) + len(self.b)
        )
        self.valid_states = self._build_valid_states()
        initial_value = int(self.a) if task == "add" else 0
        initial_checkpoint = str(initial_value).zfill(self.scratch_width)
        self.max_progress = self.valid_states.get(initial_checkpoint, 0)

    def _build_valid_states(self):
        if self.task == "add":
            contributions = [
                int(digit) * 10**place
                for place, digit in enumerate(reversed(self.b))
            ]
            base = int(self.a)
        else:
            contributions = [
                int(self.a) * int(digit) * 10**place
                for place, digit in enumerate(reversed(self.b))
            ]
            base = 0

        states = {}
        for selected in product((0, 1), repeat=len(contributions)):
            value = base + sum(
                contribution
                for use, contribution in zip(selected, contributions)
                if use
            )
            digits = str(value).zfill(self.scratch_width)
            states[digits] = max(states.get(digits, 0), sum(selected))
        return states

    def _evaluate_input(self, env):
        b_start = len(self.a) + 2
        b_end = b_start + len(self.b)
        for position, initial_value in env.initial_values.items():
            value = env.value_tape.get(position, READ_BLANK)
            role = env.role_tape.get(position, ROLE_NONE)
            if b_start <= position < b_end:
                if (value, role) not in (
                    (initial_value, ROLE_INPUT),
                    (initial_value, ROLE_USED),
                ):
                    return False, "input B was modified"
            elif value != initial_value or role != env.initial_roles[position]:
                return False, "input was modified"
        return True, "input intact"

    def is_success(self, env):
        expected_positions = {
            env.output_start + offset: read_digit(digit)
            for offset, digit in enumerate(self.answer)
        }
        for position, expected in expected_positions.items():
            if env.value_tape.get(position, READ_BLANK) != expected:
                return False
        for position, value in env.value_tape.items():
            if position >= env.output_start and position not in expected_positions:
                if value != READ_BLANK:
                    return False
        return True

    def _evaluate_output(self, env):
        if self.is_success(env):
            return self.OUTPUT_REWARD, True, "correct output"
        return 0.0, False, "output incomplete"

    def _result_digits(self, env):
        cells = sorted(
            (position, env.value_tape.get(position, READ_BLANK))
            for position, role in env.role_tape.items()
            if position < 0 and role == ROLE_RES
        )
        if not cells:
            return None
        digits = []
        for _, value in cells:
            if not (READ_DIGIT_BASE <= value < READ_DIGIT_BASE + 10):
                return False
            digits.append(str(value - READ_DIGIT_BASE))
        while len(digits) > self.scratch_width and digits[0] == "0":
            digits.pop(0)
        if len(digits) > self.scratch_width:
            return False
        return "".join(digits)

    def _evaluate_scratch(self, env):
        digits = self._result_digits(env)
        if digits is False:
            return 0.0, "scratch has an invalid RES value"
        if digits is None:
            return 0.0, "no accumulator yet"

        if len(digits) < self.scratch_width:
            return 0.0, "partial accumulator"

        progress = self.valid_states.get(digits)
        if progress is None:
            return 0.0, "scratch is not an arithmetic checkpoint"
        if progress <= self.max_progress:
            return 0.0, "known arithmetic checkpoint"

        reward = self.CHECKPOINT_REWARD * (progress - self.max_progress)
        self.max_progress = progress
        return reward, "new arithmetic checkpoint"

    def step(self, env, proposed_action):
        candidate = env.clone()
        try:
            candidate.apply(proposed_action)
        except (InvalidAction, ValueError) as exc:
            return self._intervene(env, f"mechanically invalid: {exc}")

        valid, reason = self._evaluate_input(candidate)
        if not valid:
            return self._intervene(env, reason)

        env.apply(proposed_action)
        scratch_reward, scratch_reason = self._evaluate_scratch(env)
        output_reward, success, output_reason = self._evaluate_output(env)
        reward = self.STEP_PENALTY + scratch_reward + output_reward
        return TeacherDecision(
            executed_action=tuple(proposed_action),
            reward=reward,
            intervened=False,
            success=success,
            reason=f"{scratch_reason}; {output_reason}",
            budget_cost=1,
        )

    def _intervene(self, env, reason):
        env.apply(NEUTRAL_ACTION)
        return TeacherDecision(
            executed_action=NEUTRAL_ACTION,
            reward=self.INVALID_PENALTY,
            intervened=True,
            success=self.is_success(env),
            reason=reason,
            budget_cost=1,
        )
