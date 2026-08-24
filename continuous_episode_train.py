"""Repeated-question recurrent actor-critic training for arithmetic tapes.

Stage 1 briefly behavior-clones generated arithmetic trajectories. It stops at
a configurable held-out tick accuracy (or a hard step limit), and the critic
parameters are excluded from the optimizer.

Stage 2 treats one question as a trial containing several fresh tape episodes.
An episode ends when the output becomes correct or its movement budget is
exhausted. The tape resets to the same question while the Mamba state persists.
Only the last episode terminates the trial, so discounted returns and GAE cross
episode boundaries. The previous scalar reward is an explicit input on the next
tick.

Several independent questions run in parallel. A fixed stage-2 question set is
reshuffled and revisited for multiple dataset epochs. Evaluation reports
success and tick count by episode index, both with state carry and with a fully
fresh-state control.
"""

import math
import os
import random
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from continuous_arith_data_loader import (
    ArithmeticTickDataLoader,
    ArithmeticTickDataset,
    MOVE_INPUT_VOCAB_SIZE,
    MOVE_LEFT,
    MOVE_RIGHT,
    MOVE_START,
    MOVE_STAY,
    MOVE_VOCAB_SIZE,
    READ_ADD,
    READ_BLANK,
    READ_END,
    READ_MUL,
    READ_PAD,
    READ_SEP,
    READ_VOCAB_SIZE,
    WRITE_INPUT_VOCAB_SIZE,
    WRITE_NOOP,
    WRITE_START,
    WRITE_VOCAB_SIZE,
    generate_add,
    generate_mul,
    read_digit,
    write_to_read,
)
from continuous_mamba3_model import GPT, GPTConfig


# --- Experiment configuration ------------------------------------------------
DEVICE = "cuda"
COMPILE = True
RUN_STAGE1 = True
RUN_STAGE2 = True
LOAD_CHECKPOINT = None
TASKS = ("add", "mul")
TRAIN_DIGIT_RANGES = {"add": (1, 8), "mul": (1, 4)}
TEST_LONG_DIGIT_RANGES = {"add": (12, 16), "mul": (6, 8)}

# Stage 1 intentionally stops before driving imitation loss to zero.
STAGE1_ITEMS_PER_TASK = 50_000
STAGE1_EVAL_ITEMS_PER_TASK = 500
STAGE1_MICRO_BATCH_SIZE = 128
STAGE1_GRAD_ACCUM_STEPS = 1
STAGE1_MIN_STEPS = 100
STAGE1_EVAL_INTERVAL = 25
STAGE1_STOP_TICK_ACCURACY = 0.995
STAGE1_MAX_LR = 3e-4
STAGE1_MIN_LR = STAGE1_MAX_LR * 0.1
STAGE1_WARMUP_STEPS = 30

# One update is made from each newly sampled parallel question batch.
STAGE2_QUESTION_SET_SIZE = 128
STAGE2_DATASET_EPOCHS = 20
STAGE2_PARALLEL_TRIALS = 4
STAGE2_EPISODES_PER_TRIAL = 6
STAGE2_BUDGET_FACTOR = 1.5
STAGE2_EVAL_INTERVAL_EPOCHS = 1
STAGE2_EVAL_QUESTIONS = 32
STAGE2_LR = 1e-4

SUCCESS_REWARD = 1.0
FAILURE_REWARD = -1.0
NO_REWARD = 0.0
DISCOUNT = 0.999
GAE_LAMBDA = 1.0
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.001
MAX_GRAD_NORM = 1.0

NUM_ACTION_FACTORS = 4
START_ACTION = (MOVE_START, MOVE_START, WRITE_START, WRITE_START)

config = GPTConfig(
    block_size=32,
    input_streams=(
        ("read", READ_VOCAB_SIZE),
        ("read", READ_VOCAB_SIZE),
    ),
    action_input_streams=(
        ("move", MOVE_INPUT_VOCAB_SIZE),
        ("move", MOVE_INPUT_VOCAB_SIZE),
        ("write", WRITE_INPUT_VOCAB_SIZE),
        ("write", WRITE_INPUT_VOCAB_SIZE),
    ),
    output_streams=(
        ("move", MOVE_VOCAB_SIZE),
        ("move", MOVE_VOCAB_SIZE),
        ("write", WRITE_VOCAB_SIZE),
        ("write", WRITE_VOCAB_SIZE),
    ),
    token_embd=32,
    n_layers=4,
    n_heads=4,
    n_embd=64,
    critic_outputs=1,
    use_latent=False,
    scalar_input_dim=1,
)

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "continuous_episode_mamba3_log.txt")


# --- Shared helpers -----------------------------------------------------------
def generate_decimal_number(rng, min_digits, max_digits):
    length = rng.randint(min_digits, max_digits)
    if length == 1:
        return rng.choice("0123456789")
    return rng.choice("123456789") + "".join(
        rng.choice("0123456789") for _ in range(length - 1)
    )


def generate_number_pairs(count, digit_range, seed):
    min_digits, max_digits = digit_range
    if count < 0:
        raise ValueError("item count cannot be negative")
    if min_digits <= 0 or max_digits < min_digits:
        raise ValueError(f"invalid digit range {digit_range}")
    rng = random.Random(seed)
    return [
        (
            generate_decimal_number(rng, min_digits, max_digits),
            generate_decimal_number(rng, min_digits, max_digits),
        )
        for _ in range(count)
    ]


def build_imitation_dataset(count, digit_ranges, seed, shuffle_seed=None):
    items = {
        task: generate_number_pairs(
            count,
            digit_ranges[task],
            seed=seed + task_index,
        )
        for task_index, task in enumerate(TASKS)
    }
    return ArithmeticTickDataset(
        add_items=items.get("add", []),
        mul_items=items.get("mul", []),
        tasks=TASKS,
        shuffle_seed=shuffle_seed,
    )


def is_critic_parameter(name):
    return name.startswith(("critic_mlp.", "critic_ln.", "critic_head."))


def configure_optimizer(model, learning_rate, include_critic):
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (include_critic or not is_critic_parameter(name))
    }
    decay = [parameter for parameter in parameters.values() if parameter.ndim >= 2]
    no_decay = [
        parameter for parameter in parameters.values() if parameter.ndim < 2
    ]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def stage1_learning_rate(step, total_steps):
    if step < STAGE1_WARMUP_STEPS:
        return STAGE1_MAX_LR * (step + 1) / STAGE1_WARMUP_STEPS
    if step >= total_steps:
        return STAGE1_MIN_LR
    decay_ratio = (step - STAGE1_WARMUP_STEPS) / max(
        total_steps - STAGE1_WARMUP_STEPS, 1
    )
    coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return STAGE1_MIN_LR + coefficient * (
        STAGE1_MAX_LR - STAGE1_MIN_LR
    )


def masked_mean(values, mask):
    mask = mask.to(dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def unpack_stage1_batch(batch, device):
    (
        head0_reads,
        head1_reads,
        prev_head0_moves,
        prev_head1_moves,
        prev_head0_writes,
        prev_head1_writes,
        target_head0_moves,
        target_head1_moves,
        target_head0_writes,
        target_head1_writes,
        mask,
    ) = [tensor.to(device) for tensor in batch]

    reads = torch.stack((head0_reads, head1_reads), dim=-1)
    previous_rewards = mask.new_zeros((*mask.shape, 1))
    previous_actions = torch.stack(
        (
            prev_head0_moves,
            prev_head1_moves,
            prev_head0_writes,
            prev_head1_writes,
        ),
        dim=-1,
    )
    targets = (
        target_head0_moves,
        target_head1_moves,
        target_head0_writes,
        target_head1_writes,
    )
    return reads, previous_rewards, previous_actions, targets, mask


def recurrent_forward_chunked(
    model,
    reads,
    previous_rewards,
    previous_actions,
):
    """Retain the Mamba graph across chunks for full-sequence BPTT."""
    states = None
    logits_by_factor = [[] for _ in range(config.num_outputs)]
    values = []
    for start in range(0, reads.size(1), config.block_size):
        end = min(start + config.block_size, reads.size(1))
        logits, chunk_values, states = model(
            reads[:, start:end].contiguous(),
            previous_actions[:, start:end].contiguous(),
            states=states,
            scalar_inputs=previous_rewards[:, start:end].contiguous(),
        )
        for factor, factor_logits in enumerate(logits):
            logits_by_factor[factor].append(factor_logits)
        values.append(chunk_values.squeeze(-1))
    return (
        tuple(
            torch.cat(factor_parts, dim=1)
            for factor_parts in logits_by_factor
        ),
        torch.cat(values, dim=1),
    )


def factorized_imitation_loss(logits, targets, mask):
    loss_sum = mask.new_tensor(0.0)
    factor_correct = torch.zeros_like(mask)
    tick_correct = torch.ones_like(mask, dtype=torch.bool)
    for factor_logits, target in zip(logits, targets):
        per_tick = F.cross_entropy(
            factor_logits.reshape(-1, factor_logits.size(-1)),
            target.reshape(-1),
            reduction="none",
        ).view_as(target)
        loss_sum = loss_sum + (per_tick * mask).sum()
        correct = factor_logits.argmax(dim=-1).eq(target)
        factor_correct = factor_correct + correct.float()
        tick_correct &= correct

    valid_ticks = mask.sum().clamp_min(1.0)
    return loss_sum / (valid_ticks * NUM_ACTION_FACTORS), {
        "factor_accuracy": (
            (factor_correct * mask).sum() / (valid_ticks * NUM_ACTION_FACTORS)
        ).detach(),
        "tick_accuracy": masked_mean(tick_correct.float(), mask).detach(),
    }


# --- Stage 1: deliberately limited behavior cloning --------------------------
def stage1_datasets():
    train = build_imitation_dataset(
        STAGE1_ITEMS_PER_TASK,
        TRAIN_DIGIT_RANGES,
        seed=42,
        shuffle_seed=42,
    )
    validation = build_imitation_dataset(
        STAGE1_EVAL_ITEMS_PER_TASK,
        TRAIN_DIGIT_RANGES,
        seed=123,
    )
    test = build_imitation_dataset(
        STAGE1_EVAL_ITEMS_PER_TASK,
        TEST_LONG_DIGIT_RANGES,
        seed=456,
    )
    return train, validation, test


@torch.no_grad()
def evaluate_stage1(model, loader):
    model.eval()
    totals = {"loss": 0.0, "factor_accuracy": 0.0, "tick_accuracy": 0.0}
    total_ticks = 0
    for batch in loader:
        reads, previous_rewards, previous_actions, targets, mask = (
            unpack_stage1_batch(batch, DEVICE)
        )
        sequence_length = int(mask.sum(dim=1).max().item())
        reads = reads[:, :sequence_length]
        previous_rewards = previous_rewards[:, :sequence_length]
        previous_actions = previous_actions[:, :sequence_length]
        targets = tuple(target[:, :sequence_length] for target in targets)
        active_mask = mask[:, :sequence_length]
        logits, _ = recurrent_forward_chunked(
            model,
            reads,
            previous_rewards,
            previous_actions,
        )
        loss, metrics = factorized_imitation_loss(logits, targets, active_mask)
        ticks = int(active_mask.sum().item())
        totals["loss"] += float(loss.item()) * ticks
        for name, value in metrics.items():
            totals[name] += float(value.item()) * ticks
        total_ticks += ticks
    return {
        name: value / max(total_ticks, 1)
        for name, value in totals.items()
    }


def train_stage1(model, evaluation_model):
    train_dataset, validation_dataset, test_dataset = stage1_datasets()
    train_loader = ArithmeticTickDataLoader(
        train_dataset,
        batch_size=STAGE1_MICRO_BATCH_SIZE,
        pad_to_multiple=config.block_size,
    )
    validation_loader = ArithmeticTickDataLoader(
        validation_dataset,
        batch_size=STAGE1_MICRO_BATCH_SIZE,
        pad_to_multiple=config.block_size,
    )
    test_loader = ArithmeticTickDataLoader(
        test_dataset,
        batch_size=STAGE1_MICRO_BATCH_SIZE,
        pad_to_multiple=config.block_size,
    )
    effective_batch_size = (
        STAGE1_MICRO_BATCH_SIZE * STAGE1_GRAD_ACCUM_STEPS
    )
    total_steps = len(train_dataset) // effective_batch_size
    if total_steps <= 0:
        raise ValueError(
            "stage 1 dataset must contain at least one effective batch"
        )
    optimizer = configure_optimizer(
        evaluation_model, STAGE1_MAX_LR, include_critic=False
    )
    batches = iter(train_loader)
    completed_steps = 0

    print(f"Stage 1 train: {len(train_dataset)} examples")
    print(
        f"Stage 1 micro batch: {STAGE1_MICRO_BATCH_SIZE}, "
        f"grad accum: {STAGE1_GRAD_ACCUM_STEPS}, "
        f"effective batch: {effective_batch_size}"
    )
    print(f"Stage 1 optimizer steps: {total_steps}")

    while completed_steps < total_steps:
        if completed_steps % STAGE1_EVAL_INTERVAL == 0:
            validation = evaluate_stage1(evaluation_model, validation_loader)
            message = (
                f"stage1 {completed_steps:4d}/{total_steps} val | "
                f"loss {validation['loss']:.4f} | "
                f"factor_acc {validation['factor_accuracy']:.4f} | "
                f"tick_acc {validation['tick_accuracy']:.4f}"
            )
            print(message)
            with open(log_file, "a") as log:
                log.write(message + "\n")
            if (
                completed_steps >= STAGE1_MIN_STEPS
                and validation["tick_accuracy"] >= STAGE1_STOP_TICK_ACCURACY
            ):
                print("Stage 1 stopped at the configured validation accuracy")
                break

        learning_rate = stage1_learning_rate(completed_steps, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        model.train()
        optimizer.zero_grad(set_to_none=True)
        started = time.time()
        accumulated_loss = 0.0
        accumulated_factor_accuracy = 0.0
        accumulated_tick_accuracy = 0.0
        for _ in range(STAGE1_GRAD_ACCUM_STEPS):
            batch = next(batches)
            reads, previous_rewards, previous_actions, targets, mask = (
                unpack_stage1_batch(batch, DEVICE)
            )
            sequence_length = int(mask.sum(dim=1).max().item())
            reads = reads[:, :sequence_length]
            previous_rewards = previous_rewards[:, :sequence_length]
            previous_actions = previous_actions[:, :sequence_length]
            targets = tuple(target[:, :sequence_length] for target in targets)
            active_mask = mask[:, :sequence_length]

            logits, _ = recurrent_forward_chunked(
                model,
                reads,
                previous_rewards,
                previous_actions,
            )
            loss, metrics = factorized_imitation_loss(
                logits, targets, active_mask
            )
            (loss / STAGE1_GRAD_ACCUM_STEPS).backward()
            accumulated_loss += float(loss.detach()) / STAGE1_GRAD_ACCUM_STEPS
            accumulated_factor_accuracy += (
                float(metrics["factor_accuracy"]) / STAGE1_GRAD_ACCUM_STEPS
            )
            accumulated_tick_accuracy += (
                float(metrics["tick_accuracy"]) / STAGE1_GRAD_ACCUM_STEPS
            )

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), MAX_GRAD_NORM
        )
        optimizer.step()
        completed_steps += 1
        print(
            f"stage1 {completed_steps:4d}/{total_steps} | "
            f"loss {accumulated_loss:.4f} | "
            f"factor_acc {accumulated_factor_accuracy:.4f} | "
            f"tick_acc {accumulated_tick_accuracy:.4f} | "
            f"norm {float(gradient_norm):.4f} | "
            f"lr {learning_rate:.2e} | dt {time.time() - started:.2f}s"
        )

    validation = evaluate_stage1(evaluation_model, validation_loader)
    test = evaluate_stage1(evaluation_model, test_loader)
    for label, result in (("val", validation), ("test-long", test)):
        message = (
            f"stage1 final {label} | loss {result['loss']:.4f} | "
            f"factor_acc {result['factor_accuracy']:.4f} | "
            f"tick_acc {result['tick_accuracy']:.4f}"
        )
        print(message)
        with open(log_file, "a") as log:
            log.write(message + "\n")


# --- Tape environment --------------------------------------------------------
def move_delta(move):
    if move == MOVE_LEFT:
        return -1
    if move == MOVE_RIGHT:
        return 1
    if move == MOVE_STAY:
        return 0
    raise ValueError(f"invalid move token {move}")


class ArithmeticTapeEnvironment:
    def __init__(self, task, a, b):
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}")
        self.task = task
        self.a = str(a)
        self.b = str(b)
        operation = READ_ADD if task == "add" else READ_MUL
        initial = (
            [operation]
            + [read_digit(digit) for digit in self.a]
            + [READ_SEP]
            + [read_digit(digit) for digit in self.b]
            + [READ_END]
        )
        self.tape = {position: token for position, token in enumerate(initial)}
        self.output_start = len(initial)
        self.answer = str(
            int(self.a) + int(self.b)
            if task == "add"
            else int(self.a) * int(self.b)
        )
        self.head0 = 0
        self.head1 = 0
        self.previous_action = START_ACTION

    def reads(self):
        return (
            self.tape.get(self.head0, READ_BLANK),
            self.tape.get(self.head1, READ_BLANK),
        )

    def apply(self, action):
        action = tuple(int(token) for token in action)
        if len(action) != NUM_ACTION_FACTORS:
            raise ValueError(
                f"expected {NUM_ACTION_FACTORS} action factors, got {len(action)}"
            )
        head0_move, head1_move, head0_write, head1_write = action
        for move in (head0_move, head1_move):
            if not (MOVE_STAY <= move < MOVE_VOCAB_SIZE):
                raise ValueError(f"invalid move token {move}")
        for write in (head0_write, head1_write):
            if not (WRITE_NOOP <= write < WRITE_VOCAB_SIZE):
                raise ValueError(f"invalid write token {write}")

        # Head 1 wins when both heads write the same cell in one tick.
        for position, write in (
            (self.head0, head0_write),
            (self.head1, head1_write),
        ):
            token = write_to_read(write)
            if token is None:
                continue
            if token == READ_BLANK:
                self.tape.pop(position, None)
            else:
                self.tape[position] = token
        self.head0 += move_delta(head0_move)
        self.head1 += move_delta(head1_move)
        self.previous_action = action

    def is_success(self):
        expected_positions = {
            self.output_start + offset: read_digit(digit)
            for offset, digit in enumerate(self.answer)
        }
        for position, expected in expected_positions.items():
            if self.tape.get(position, READ_BLANK) != expected:
                return False
        for position, token in self.tape.items():
            if (
                position >= self.output_start
                and position not in expected_positions
                and token != READ_BLANK
            ):
                return False
        return True


def reference_ticks(task, a, b):
    trajectory = generate_add(a, b) if task == "add" else generate_mul(a, b)
    return len(trajectory.target_head0_moves)


# --- Repeated-question trials ------------------------------------------------
@dataclass
class QuestionTrial:
    task: str
    a: str
    b: str
    budget: int
    environment: ArithmeticTapeEnvironment
    previous_reward: float = NO_REWARD
    episode_index: int = 0
    episode_steps: int = 0
    done: bool = False

    def finish_episode(self, reward):
        self.episode_index += 1
        self.episode_steps = 0
        if self.episode_index >= STAGE2_EPISODES_PER_TRIAL:
            self.done = True
            return
        self.environment = ArithmeticTapeEnvironment(self.task, self.a, self.b)
        self.previous_reward = reward


@dataclass
class TrialTrajectory:
    reads: torch.Tensor
    previous_rewards: torch.Tensor
    previous_actions: torch.Tensor
    actions: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None

    @property
    def num_ticks(self):
        return self.rewards.numel()


def initialize_question_trials(items):
    trials = []
    for task, a, b in items:
        a = str(a)
        b = str(b)
        trials.append(
            QuestionTrial(
                task=task,
                a=a,
                b=b,
                budget=max(
                    1,
                    math.ceil(
                        reference_ticks(task, a, b) * STAGE2_BUDGET_FACTOR
                    ),
                ),
                environment=ArithmeticTapeEnvironment(task, a, b),
            )
        )
    return trials


def step_trial_episode(trial, action):
    """Apply one action and return (episode_done, succeeded, reward)."""
    trial.episode_steps += 1
    trial.environment.apply(action)
    if trial.environment.is_success():
        return True, True, SUCCESS_REWARD
    if trial.episode_steps >= trial.budget:
        return True, False, FAILURE_REWARD
    return False, False, NO_REWARD


def sample_factorized_actions(logits):
    actions = []
    entropies = []
    for factor_logits in logits:
        factor_logits = factor_logits[:, 0]
        log_probs = F.log_softmax(factor_logits, dim=-1)
        probabilities = log_probs.exp()
        actions.append(
            torch.multinomial(probabilities, num_samples=1).squeeze(-1)
        )
        entropies.append(-(probabilities * log_probs).sum(dim=-1))
    return torch.stack(actions, dim=-1), torch.stack(entropies, dim=-1).sum(-1)


def compute_trial_gae(trajectory):
    advantages = torch.zeros_like(trajectory.values)
    gae = trajectory.values.new_tensor(0.0)
    next_value = trajectory.values.new_tensor(0.0)
    for tick in range(trajectory.num_ticks - 1, -1, -1):
        nonterminal = 1.0 - trajectory.dones[tick]
        delta = (
            trajectory.rewards[tick]
            + DISCOUNT * next_value * nonterminal
            - trajectory.values[tick]
        )
        gae = delta + DISCOUNT * GAE_LAMBDA * nonterminal * gae
        advantages[tick] = gae
        next_value = trajectory.values[tick]
    trajectory.advantages = advantages
    trajectory.returns = advantages + trajectory.values


@torch.no_grad()
def collect_trial_batch(model, items):
    model.eval()
    trials = initialize_question_trials(items)
    buffers = [
        {
            "reads": [],
            "previous_rewards": [],
            "previous_actions": [],
            "actions": [],
            "values": [],
            "rewards": [],
            "dones": [],
        }
        for _ in trials
    ]
    successes = [0] * STAGE2_EPISODES_PER_TRIAL
    ticks = [0] * STAGE2_EPISODES_PER_TRIAL
    success_ticks = [0] * STAGE2_EPISODES_PER_TRIAL
    tick_records = [
        [0] * STAGE2_EPISODES_PER_TRIAL for _ in trials
    ]
    states = None

    while any(not trial.done for trial in trials):
        read_rows = []
        previous_reward_rows = []
        previous_rows = []
        for trial in trials:
            if trial.done:
                read_rows.append((READ_PAD, READ_PAD))
                previous_reward_rows.append((NO_REWARD,))
                previous_rows.append(
                    (MOVE_STAY, MOVE_STAY, WRITE_NOOP, WRITE_NOOP)
                )
            else:
                head0_read, head1_read = trial.environment.reads()
                read_rows.append((head0_read, head1_read))
                previous_reward_rows.append((trial.previous_reward,))
                previous_rows.append(trial.environment.previous_action)

        reads = torch.tensor(
            read_rows, dtype=torch.long, device=DEVICE
        ).unsqueeze(1)
        previous_actions = torch.tensor(
            previous_rows, dtype=torch.long, device=DEVICE
        ).unsqueeze(1)
        previous_rewards = torch.tensor(
            previous_reward_rows, dtype=torch.float32, device=DEVICE
        ).unsqueeze(1)
        logits, values, states = model(
            reads,
            previous_actions,
            states=states,
            scalar_inputs=previous_rewards,
        )
        actions, _ = sample_factorized_actions(logits)
        action_rows = actions.tolist()

        for index, trial in enumerate(trials):
            if trial.done:
                continue
            episode_index = trial.episode_index
            action = tuple(action_rows[index])
            episode_done, succeeded, reward = step_trial_episode(trial, action)
            if episode_done:
                successes[episode_index] += int(succeeded)
                ticks[episode_index] += trial.episode_steps
                tick_records[index][episode_index] = trial.episode_steps
                if succeeded:
                    success_ticks[episode_index] += trial.episode_steps

            buffer = buffers[index]
            buffer["reads"].append(read_rows[index])
            buffer["previous_rewards"].append(
                previous_reward_rows[index][0]
            )
            buffer["previous_actions"].append(previous_rows[index])
            buffer["actions"].append(action)
            buffer["values"].append(float(values[index, 0, 0].item()))
            buffer["rewards"].append(reward)

            if episode_done:
                trial.finish_episode(reward)
            else:
                trial.previous_reward = NO_REWARD
            buffer["dones"].append(float(trial.done))

    trajectories = []
    for buffer in buffers:
        trajectory = TrialTrajectory(
            reads=torch.tensor(buffer["reads"], dtype=torch.long),
            previous_rewards=torch.tensor(
                buffer["previous_rewards"], dtype=torch.float32
            ).unsqueeze(-1),
            previous_actions=torch.tensor(
                buffer["previous_actions"], dtype=torch.long
            ),
            actions=torch.tensor(buffer["actions"], dtype=torch.long),
            values=torch.tensor(buffer["values"], dtype=torch.float32),
            rewards=torch.tensor(buffer["rewards"], dtype=torch.float32),
            dones=torch.tensor(buffer["dones"], dtype=torch.float32),
        )
        compute_trial_gae(trajectory)
        trajectories.append(trajectory)

    trial_count = len(trials)
    return trajectories, {
        "successes": successes,
        "ticks": ticks,
        "success_ticks": success_ticks,
        "tick_records": tick_records,
        "counts": [trial_count] * STAGE2_EPISODES_PER_TRIAL,
        "reward": sum(float(t.rewards.sum().item()) for t in trajectories),
        "total_ticks": sum(t.num_ticks for t in trajectories),
    }


def normalize_advantages(trajectories):
    all_advantages = torch.cat(
        [trajectory.advantages for trajectory in trajectories]
    )
    mean = all_advantages.mean()
    standard_deviation = all_advantages.std(unbiased=False).clamp_min(1e-8)
    for trajectory in trajectories:
        trajectory.advantages = (
            trajectory.advantages - mean
        ) / standard_deviation


def collate_trajectories(trajectories):
    batch_size = len(trajectories)
    max_ticks = max(trajectory.num_ticks for trajectory in trajectories)
    padded_ticks = (
        (max_ticks + config.block_size - 1) // config.block_size
    ) * config.block_size
    reads = torch.full(
        (batch_size, padded_ticks, config.num_inputs),
        READ_PAD,
        dtype=torch.long,
    )
    previous_rewards = torch.zeros(
        batch_size,
        padded_ticks,
        config.scalar_input_dim,
        dtype=torch.float32,
    )
    previous_actions = torch.empty(
        batch_size, padded_ticks, NUM_ACTION_FACTORS, dtype=torch.long
    )
    previous_actions[..., :2] = MOVE_STAY
    previous_actions[..., 2:] = WRITE_NOOP
    actions = previous_actions.clone()
    returns = torch.zeros(batch_size, padded_ticks)
    advantages = torch.zeros(batch_size, padded_ticks)
    mask = torch.zeros(batch_size, padded_ticks)

    for row, trajectory in enumerate(trajectories):
        length = trajectory.num_ticks
        reads[row, :length] = trajectory.reads
        previous_rewards[row, :length] = trajectory.previous_rewards
        previous_actions[row, :length] = trajectory.previous_actions
        actions[row, :length] = trajectory.actions
        returns[row, :length] = trajectory.returns
        advantages[row, :length] = trajectory.advantages
        mask[row, :length] = 1.0

    return tuple(
        tensor.to(DEVICE)
        for tensor in (
            reads,
            previous_rewards,
            previous_actions,
            actions,
            returns,
            advantages,
            mask,
        )
    )


def fixed_action_statistics(logits, actions):
    log_probs = []
    entropies = []
    for factor, factor_logits in enumerate(logits):
        factor_log_probs = F.log_softmax(factor_logits, dim=-1)
        probabilities = factor_log_probs.exp()
        factor_actions = actions[..., factor]
        log_probs.append(
            factor_log_probs.gather(
                -1, factor_actions.unsqueeze(-1)
            ).squeeze(-1)
        )
        entropies.append(-(probabilities * factor_log_probs).sum(dim=-1))
    return (
        torch.stack(log_probs, dim=-1).sum(dim=-1),
        torch.stack(entropies, dim=-1).sum(dim=-1),
    )


def actor_critic_update(model, optimizer, trajectories):
    normalize_advantages(trajectories)
    (
        reads,
        previous_rewards,
        previous_actions,
        actions,
        returns,
        advantages,
        mask,
    ) = collate_trajectories(trajectories)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, values = recurrent_forward_chunked(
        model,
        reads,
        previous_rewards,
        previous_actions,
    )
    log_probs, entropies = fixed_action_statistics(logits, actions)
    policy_loss = masked_mean(-log_probs * advantages, mask)
    value_loss = 0.5 * masked_mean((values - returns).square(), mask)
    entropy = masked_mean(entropies, mask)
    loss = policy_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF * entropy
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), MAX_GRAD_NORM
    )
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "policy": float(policy_loss.detach()),
        "value": float(value_loss.detach()),
        "entropy": float(entropy.detach()),
        "gradient_norm": float(gradient_norm),
    }


# --- Evaluation ---------------------------------------------------------------
def reset_state_rows(states, reset_rows):
    if states is None or not reset_rows.any():
        return states
    keep = (~reset_rows).to(dtype=states[0][0].dtype)
    reset_states = []
    for state in states:
        reset_components = []
        for component in state:
            shape = (component.size(0),) + (1,) * (component.ndim - 1)
            reset_components.append(component * keep.view(shape))
        reset_states.append(tuple(reset_components))
    return reset_states


@torch.no_grad()
def evaluate_trial_batch(model, items, carry_state):
    model.eval()
    trials = initialize_question_trials(items)
    successes = [0] * STAGE2_EPISODES_PER_TRIAL
    ticks = [0] * STAGE2_EPISODES_PER_TRIAL
    success_ticks = [0] * STAGE2_EPISODES_PER_TRIAL
    tick_records = [
        [0] * STAGE2_EPISODES_PER_TRIAL for _ in trials
    ]
    states = None

    while any(not trial.done for trial in trials):
        read_rows = []
        previous_reward_rows = []
        previous_rows = []
        for trial in trials:
            if trial.done:
                read_rows.append((READ_PAD, READ_PAD))
                previous_reward_rows.append((NO_REWARD,))
                previous_rows.append(
                    (MOVE_STAY, MOVE_STAY, WRITE_NOOP, WRITE_NOOP)
                )
            else:
                head0_read, head1_read = trial.environment.reads()
                read_rows.append((head0_read, head1_read))
                previous_reward_rows.append((trial.previous_reward,))
                previous_rows.append(trial.environment.previous_action)

        reads = torch.tensor(
            read_rows, dtype=torch.long, device=DEVICE
        ).unsqueeze(1)
        previous_actions = torch.tensor(
            previous_rows, dtype=torch.long, device=DEVICE
        ).unsqueeze(1)
        previous_rewards = torch.tensor(
            previous_reward_rows, dtype=torch.float32, device=DEVICE
        ).unsqueeze(1)
        logits, _, states = model(
            reads,
            previous_actions,
            states=states,
            scalar_inputs=previous_rewards,
        )
        actions = torch.stack(
            [
                factor_logits[:, 0].argmax(dim=-1)
                for factor_logits in logits
            ],
            dim=-1,
        ).tolist()
        reset_rows = torch.zeros(len(trials), dtype=torch.bool, device=DEVICE)

        for index, trial in enumerate(trials):
            if trial.done:
                continue
            episode_index = trial.episode_index
            episode_done, succeeded, reward = step_trial_episode(
                trial,
                tuple(actions[index]),
            )

            if episode_done:
                successes[episode_index] += int(succeeded)
                ticks[episode_index] += trial.episode_steps
                tick_records[index][episode_index] = trial.episode_steps
                if succeeded:
                    success_ticks[episode_index] += trial.episode_steps
                trial.finish_episode(reward)
                if not trial.done and not carry_state:
                    trial.previous_reward = NO_REWARD
                    reset_rows[index] = True
            else:
                trial.previous_reward = NO_REWARD

        if not carry_state:
            states = reset_state_rows(states, reset_rows)

    return {
        "successes": successes,
        "ticks": ticks,
        "success_ticks": success_ticks,
        "tick_records": tick_records,
        "counts": [len(items)] * STAGE2_EPISODES_PER_TRIAL,
    }


def merge_episode_statistics(total, update):
    if total is None:
        merged = {
            name: list(values)
            for name, values in update.items()
            if name in ("successes", "ticks", "success_ticks", "counts")
        }
        merged["tick_records"] = [
            list(record) for record in update["tick_records"]
        ]
        return merged
    for name in ("successes", "ticks", "success_ticks", "counts"):
        total[name] = [
            old + new for old, new in zip(total[name], update[name])
        ]
    total["tick_records"].extend(
        list(record) for record in update["tick_records"]
    )
    return total


def finalize_episode_statistics(statistics):
    tick_records = statistics["tick_records"]
    paired_tick_deltas = [
        sum(record[episode] - record[0] for record in tick_records)
        / max(len(tick_records), 1)
        for episode in range(STAGE2_EPISODES_PER_TRIAL)
    ]
    return {
        "success": [
            successes / max(count, 1)
            for successes, count in zip(
                statistics["successes"], statistics["counts"]
            )
        ],
        "ticks": [
            ticks / max(count, 1)
            for ticks, count in zip(
                statistics["ticks"], statistics["counts"]
            )
        ],
        "success_ticks": [
            ticks / successes if successes > 0 else float("nan")
            for ticks, successes in zip(
                statistics["success_ticks"], statistics["successes"]
            )
        ],
        "paired_tick_delta": paired_tick_deltas,
    }


@torch.no_grad()
def evaluate_trials(model, items, carry_state):
    totals = None
    for start in range(0, len(items), STAGE2_PARALLEL_TRIALS):
        statistics = evaluate_trial_batch(
            model,
            items[start : start + STAGE2_PARALLEL_TRIALS],
            carry_state=carry_state,
        )
        totals = merge_episode_statistics(totals, statistics)
    return finalize_episode_statistics(totals)


def format_episode_statistics(label, statistics):
    parts = []
    for episode, (success, ticks, success_ticks, paired_delta) in enumerate(
        zip(
            statistics["success"],
            statistics["ticks"],
            statistics["success_ticks"],
            statistics["paired_tick_delta"],
        ),
        start=1,
    ):
        parts.append(
            f"e{episode} success {success:.3f} ticks {ticks:.1f} "
            f"paired_delta {paired_delta:+.1f} "
            f"solved_ticks {success_ticks:.1f}"
        )
    return f"{label} | " + " | ".join(parts)


# --- Stage 2 training ---------------------------------------------------------
def build_question_set(count, digit_ranges, seed):
    if not TASKS:
        raise ValueError("at least one task is required")
    rng = random.Random(seed)
    per_task, remainder = divmod(count, len(TASKS))
    questions = []
    for task_index, task in enumerate(TASKS):
        task_count = per_task + int(task_index < remainder)
        pairs = generate_number_pairs(
            task_count,
            digit_ranges[task],
            seed=seed + task_index + 1,
        )
        questions.extend((task, a, b) for a, b in pairs)
    rng.shuffle(questions)
    return questions


def train_stage2(model, evaluation_model):
    train_questions = build_question_set(
        STAGE2_QUESTION_SET_SIZE,
        TRAIN_DIGIT_RANGES,
        seed=2026,
    )
    validation_questions = build_question_set(
        STAGE2_EVAL_QUESTIONS,
        TRAIN_DIGIT_RANGES,
        seed=2027,
    )
    test_questions = build_question_set(
        STAGE2_EVAL_QUESTIONS,
        TEST_LONG_DIGIT_RANGES,
        seed=2028,
    )
    optimizer = configure_optimizer(
        evaluation_model, STAGE2_LR, include_critic=True
    )
    rng = random.Random(2029)
    total_updates = (
        math.ceil(len(train_questions) / STAGE2_PARALLEL_TRIALS)
        * STAGE2_DATASET_EPOCHS
    )
    completed_updates = 0

    for epoch in range(STAGE2_DATASET_EPOCHS):
        if epoch % STAGE2_EVAL_INTERVAL_EPOCHS == 0:
            carry = evaluate_trials(
                evaluation_model, validation_questions, carry_state=True
            )
            fresh = evaluate_trials(
                evaluation_model, validation_questions, carry_state=False
            )
            for message in (
                format_episode_statistics(
                    f"stage2 epoch {epoch} val-carry", carry
                ),
                format_episode_statistics(
                    f"stage2 epoch {epoch} val-fresh", fresh
                ),
            ):
                print(message)
                with open(log_file, "a") as log:
                    log.write(message + "\n")

        rng.shuffle(train_questions)
        epoch_statistics = None
        for start in range(0, len(train_questions), STAGE2_PARALLEL_TRIALS):
            batch_items = train_questions[
                start : start + STAGE2_PARALLEL_TRIALS
            ]
            started = time.time()
            trajectories, rollout = collect_trial_batch(
                evaluation_model, batch_items
            )
            metrics = actor_critic_update(model, optimizer, trajectories)
            completed_updates += 1
            epoch_statistics = merge_episode_statistics(
                epoch_statistics, rollout
            )
            print(
                f"stage2 step {completed_updates:5d}/{total_updates} | "
                f"epoch {epoch + 1}/{STAGE2_DATASET_EPOCHS} | "
                f"loss {metrics['loss']:.4f} | "
                f"policy {metrics['policy']:.4f} | "
                f"value {metrics['value']:.4f} | "
                f"entropy {metrics['entropy']:.4f} | "
                f"norm {metrics['gradient_norm']:.4f} | "
                f"dt {time.time() - started:.2f}s"
            )

        epoch_result = finalize_episode_statistics(epoch_statistics)
        message = format_episode_statistics(
            f"stage2 epoch {epoch + 1} train", epoch_result
        )
        print(message)
        with open(log_file, "a") as log:
            log.write(message + "\n")

    for label, items in (
        ("train", train_questions[:STAGE2_EVAL_QUESTIONS]),
        ("val", validation_questions),
        ("test-long", test_questions),
    ):
        carry = evaluate_trials(evaluation_model, items, carry_state=True)
        fresh = evaluate_trials(evaluation_model, items, carry_state=False)
        for message in (
            format_episode_statistics(f"stage2 final {label}-carry", carry),
            format_episode_statistics(f"stage2 final {label}-fresh", fresh),
        ):
            print(message)
            with open(log_file, "a") as log:
                log.write(message + "\n")


def save_checkpoint(path, model, stage):
    torch.save(
        {"model": model.state_dict(), "config": config, "stage": stage},
        path,
    )
    print(f"Saved {stage} checkpoint to {path}")


def main():
    if config.critic_outputs != 1:
        raise ValueError("repeated-question training requires a scalar critic")
    if config.scalar_input_dim != 1:
        raise ValueError("repeated-question training requires one reward input")
    if STAGE2_EPISODES_PER_TRIAL <= 0:
        raise ValueError("STAGE2_EPISODES_PER_TRIAL must be positive")
    if STAGE2_PARALLEL_TRIALS <= 0:
        raise ValueError("STAGE2_PARALLEL_TRIALS must be positive")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    raw_model = GPT(config).to(DEVICE)
    if LOAD_CHECKPOINT is not None:
        checkpoint = torch.load(LOAD_CHECKPOINT, map_location=DEVICE)
        raw_model.load_state_dict(checkpoint["model"])
    model = torch.compile(raw_model) if COMPILE else raw_model

    parameters = sum(parameter.numel() for parameter in raw_model.parameters())
    print(f"Model: {parameters:,} parameters")
    print(f"Config: {config}")
    print(f"Compile: {COMPILE} | device: {DEVICE}")
    print(f"Episodes per trial: {STAGE2_EPISODES_PER_TRIAL}")
    print(f"Parallel trials: {STAGE2_PARALLEL_TRIALS}")
    print(f"Stage 2 dataset epochs: {STAGE2_DATASET_EPOCHS}")
    print("Latent input/output: none")
    print("Previous reward input: raw scalar")
    print("Episode termination: automatic success or budget")
    print("Policy sentinels: none; START is previous-action input only")
    print("Mamba BPTT: full trial")

    with open(log_file, "w"):
        pass
    if RUN_STAGE1:
        train_stage1(model, raw_model)
        save_checkpoint(
            os.path.join(log_dir, "continuous_episode_stage1_model.pt"),
            raw_model,
            "stage1",
        )
    if RUN_STAGE2:
        train_stage2(model, raw_model)
        save_checkpoint(
            os.path.join(log_dir, "continuous_episode_stage2_model.pt"),
            raw_model,
            "stage2",
        )


if __name__ == "__main__":
    main()
