"""Two-stage role-channel arithmetic RL training.

Stage 1 is the existing parallel teacher-state contextual-bandit objective with
random latent inputs. Stage 2 collects complete frozen-policy trajectories,
then performs shuffled off-policy sequence replay with Mamba state and no
latent feedback. No THINK or HALT action is introduced.
"""

from dataclasses import dataclass
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from tick_rl_mamba3_model import GPT, GPTConfig
from tick_role_arith_data_loader import (
    INPUT_STREAMS,
    MOVE_STAY,
    OUTPUT_STREAMS,
    ROLE_WRITE_NOOP,
    TEST_LONG_ADD,
    TEST_LONG_MUL,
    TRAIN_ADD,
    TRAIN_MUL,
    VAL_ADD,
    VAL_MUL,
    WRITE_NOOP,
    RoleArithmeticTickDataLoader,
    RoleArithmeticTickDataset,
    generate_add,
    generate_mul,
    unpack_role_batch,
)
from tick_role_arith_teacher import (
    NEUTRAL_ACTION,
    NUM_ACTION_FACTORS,
    InvalidAction,
    RoleAwareTeacher,
    RoleTapeEnvironment,
)


DEVICE = "cuda"
COMPILE = True
RUN_STAGE2 = False
TASKS = ("add", "mul")

# Stage 1 keeps the original parallel dense-RL setup.
STAGE1_BATCH_SIZE = 128
STAGE1_GRAD_ACCUM_STEPS = 1
STAGE1_EVAL_INTERVAL = 50
STAGE1_REWARD_CORRECT = 1.0
STAGE1_REWARD_INCORRECT = -1.0
STAGE1_VALUE_COEF = 0.5
STAGE1_ENTROPY_COEF = 0.01
STAGE1_MAX_LR = 3e-4
STAGE1_MIN_LR = STAGE1_MAX_LR * 0.1
STAGE1_WARMUP_STEPS = 30

# Stage 2 alternates frozen-policy collection with trajectory replay.
STAGE2_TRAINING_CYCLES = 100
STAGE2_ROLLOUTS_PER_CYCLE = 1024
STAGE2_ROLLOUT_BATCH_SIZE = 128
STAGE2_REPLAY_BATCH_SIZE = 128
STAGE2_REPLAY_EPOCHS_PER_CYCLE = 4
STAGE2_EVAL_INTERVAL_CYCLES = 5
STAGE2_LR = 1e-4
STAGE2_VALUE_COEF = 0.5
STAGE2_ENTROPY_COEF = 0.005
STAGE2_BUDGET_FACTOR = 1.5

config = GPTConfig(
    block_size=32,
    input_streams=INPUT_STREAMS,
    output_streams=OUTPUT_STREAMS,
    token_embd=32,
    n_layers=4,
    n_heads=4,
    n_embd=64,
)

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "tick_arith_two_stage_rl_mamba3_log.txt")


def masked_mean(values, mask):
    mask = mask.to(dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def configure_optimizer(model, learning_rate, weight_decay=0.1):
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    decay = [parameter for parameter in parameters.values() if parameter.ndim >= 2]
    no_decay = [parameter for parameter in parameters.values() if parameter.ndim < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def stage1_lr(step, total_steps):
    if step < STAGE1_WARMUP_STEPS:
        return STAGE1_MAX_LR * (step + 1) / STAGE1_WARMUP_STEPS
    if step >= total_steps:
        return STAGE1_MIN_LR
    ratio = (step - STAGE1_WARMUP_STEPS) / max(
        total_steps - STAGE1_WARMUP_STEPS, 1
    )
    coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return STAGE1_MIN_LR + coefficient * (
        STAGE1_MAX_LR - STAGE1_MIN_LR
    )


def sample_categorical(logits):
    log_probs = F.log_softmax(logits, dim=-1)
    probabilities = log_probs.exp()
    sampled = torch.multinomial(
        probabilities.reshape(-1, probabilities.size(-1)),
        num_samples=1,
    ).view(logits.shape[:-1])
    sampled_log_probs = log_probs.gather(
        -1, sampled.unsqueeze(-1)
    ).squeeze(-1)
    entropy = -(probabilities * log_probs).sum(dim=-1)
    return sampled, sampled_log_probs, entropy


def dense_actor_critic_loss(logits, values, targets, loss_mask):
    num_factors = len(logits)
    assert num_factors == len(targets) == config.num_outputs
    assert values.shape == (*loss_mask.shape, num_factors)

    sampled_actions = []
    sampled_log_probs = []
    entropies = []
    rewards = []
    for factor_logits, target in zip(logits, targets):
        sampled, log_prob, entropy = sample_categorical(factor_logits)
        reward = torch.where(
            sampled.eq(target),
            sampled.new_tensor(STAGE1_REWARD_CORRECT, dtype=torch.float32),
            sampled.new_tensor(STAGE1_REWARD_INCORRECT, dtype=torch.float32),
        )
        sampled_actions.append(sampled)
        sampled_log_probs.append(log_prob)
        entropies.append(entropy)
        rewards.append(reward)

    actions = torch.stack(sampled_actions, dim=-1)
    log_probs = torch.stack(sampled_log_probs, dim=-1)
    entropies = torch.stack(entropies, dim=-1)
    rewards = torch.stack(rewards, dim=-1).to(values.dtype)
    factor_mask = loss_mask.unsqueeze(-1).expand_as(values)
    advantages = (rewards - values).detach()
    policy_loss = masked_mean(-log_probs * advantages, factor_mask)
    value_loss = masked_mean((values - rewards).square(), factor_mask)
    entropy = masked_mean(entropies, factor_mask)
    loss = (
        policy_loss
        + STAGE1_VALUE_COEF * value_loss
        - STAGE1_ENTROPY_COEF * entropy
    )

    target_actions = torch.stack(targets, dim=-1)
    factor_correct = actions.eq(target_actions)
    return loss, {
        "policy": policy_loss.detach(),
        "value": value_loss.detach(),
        "entropy": entropy.detach(),
        "reward": masked_mean(rewards, factor_mask).detach(),
        "factor_acc": masked_mean(
            factor_correct.float(), factor_mask
        ).detach(),
        "tick_acc": masked_mean(
            factor_correct.all(dim=-1).float(), loss_mask
        ).detach(),
    }


def stage1_forward_chunked(model, reads, prev_actions, targets, loss_mask):
    batch_size, sequence_length = loss_mask.shape
    states = None
    weighted_loss = loss_mask.new_tensor(0.0)
    totals = None
    total_ticks = 0

    for start in range(0, sequence_length, config.block_size):
        end = min(start + config.block_size, sequence_length)
        latent = torch.randn(
            batch_size,
            end - start,
            config.n_embd,
            device=loss_mask.device,
        )
        logits, values, _, states = model(
            reads[:, start:end].contiguous(),
            prev_actions[:, start:end].contiguous(),
            latent=latent,
            states=states,
        )
        chunk_mask = loss_mask[:, start:end].contiguous()
        valid_ticks = int(chunk_mask.sum().item())
        if valid_ticks == 0:
            continue
        chunk_targets = tuple(
            target[:, start:end].contiguous() for target in targets
        )
        loss, metrics = dense_actor_critic_loss(
            logits, values, chunk_targets, chunk_mask
        )
        weighted_loss = weighted_loss + loss * valid_ticks
        if totals is None:
            totals = {
                name: metric * valid_ticks for name, metric in metrics.items()
            }
        else:
            for name, metric in metrics.items():
                totals[name] += metric * valid_ticks
        total_ticks += valid_ticks

    if total_ticks == 0:
        zero = weighted_loss.detach()
        return weighted_loss, {
            name: zero
            for name in (
                "policy",
                "value",
                "entropy",
                "reward",
                "factor_acc",
                "tick_acc",
            )
        }
    return weighted_loss / total_ticks, {
        name: total / total_ticks for name, total in totals.items()
    }


def factorized_nll(logits, targets, mask):
    losses = []
    for factor_logits, target in zip(logits, targets):
        per_tick = F.cross_entropy(
            factor_logits.reshape(-1, factor_logits.size(-1)),
            target.reshape(-1),
            reduction="none",
        ).view_as(target)
        losses.append(masked_mean(per_tick, mask))
    return sum(losses) / len(losses)


@torch.no_grad()
def evaluate_stage1(model, loader):
    model.eval()
    total_nll = 0.0
    total_ticks = 0
    factor_correct = 0.0
    tick_correct = 0.0

    for batch in loader:
        reads, prev_actions, targets, mask = unpack_role_batch(batch, DEVICE)
        batch_size, sequence_length = mask.shape
        states = None
        parts = [[] for _ in targets]
        for start in range(0, sequence_length, config.block_size):
            end = min(start + config.block_size, sequence_length)
            latent = torch.randn(
                batch_size,
                end - start,
                config.n_embd,
                device=mask.device,
            )
            logits, _, _, states = model(
                reads[:, start:end].contiguous(),
                prev_actions[:, start:end].contiguous(),
                latent=latent,
                states=states,
            )
            for factor, factor_logits in enumerate(logits):
                parts[factor].append(factor_logits)
        logits = tuple(torch.cat(factor_parts, dim=1) for factor_parts in parts)
        valid_ticks = int(mask.sum().item())
        total_nll += factorized_nll(logits, targets, mask).item() * valid_ticks
        total_ticks += valid_ticks
        correct = torch.stack(
            [
                factor_logits.argmax(dim=-1).eq(target)
                for factor_logits, target in zip(logits, targets)
            ],
            dim=-1,
        )
        factor_mask = mask.unsqueeze(-1).expand_as(correct)
        factor_correct += (correct.float() * factor_mask).sum().item()
        tick_correct += (
            correct.all(dim=-1).float() * mask
        ).sum().item()

    return {
        "nll": total_nll / total_ticks,
        "factor_acc": factor_correct / (total_ticks * config.num_outputs),
        "tick_acc": tick_correct / total_ticks,
    }


def train_stage1(model, eval_model, optimizer):
    train_dataset = RoleArithmeticTickDataset(
        add_items=TRAIN_ADD,
        mul_items=TRAIN_MUL,
        tasks=TASKS,
        shuffle_seed=42,
    )
    val_dataset = RoleArithmeticTickDataset(
        add_items=VAL_ADD,
        mul_items=VAL_MUL,
        tasks=TASKS,
    )
    test_dataset = RoleArithmeticTickDataset(
        add_items=TEST_LONG_ADD,
        mul_items=TEST_LONG_MUL,
        tasks=TASKS,
    )
    train_loader = RoleArithmeticTickDataLoader(
        train_dataset,
        batch_size=STAGE1_BATCH_SIZE,
        pad_to_multiple=config.block_size,
    )
    val_loader = RoleArithmeticTickDataLoader(
        val_dataset,
        batch_size=STAGE1_BATCH_SIZE,
        pad_to_multiple=config.block_size,
    )
    test_loader = RoleArithmeticTickDataLoader(
        test_dataset,
        batch_size=STAGE1_BATCH_SIZE,
        pad_to_multiple=config.block_size,
    )
    total_steps = len(train_loader) // STAGE1_GRAD_ACCUM_STEPS
    batches = iter(train_loader)

    for step in range(total_steps):
        if step % STAGE1_EVAL_INTERVAL == 0:
            validation = evaluate_stage1(eval_model, val_loader)
            print(
                f"stage1 {step:4d}/{total_steps} | "
                f"val_nll {validation['nll']:.4f} | "
                f"factor_acc {validation['factor_acc']:.4f} | "
                f"tick_acc {validation['tick_acc']:.4f}"
            )

        learning_rate = stage1_lr(step, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        optimizer.zero_grad()
        accumulated = None
        start_time = time.time()

        for _ in range(STAGE1_GRAD_ACCUM_STEPS):
            batch = next(batches)
            reads, prev_actions, targets, mask = unpack_role_batch(batch, DEVICE)
            loss, metrics = stage1_forward_chunked(
                model, reads, prev_actions, targets, mask
            )
            (loss / STAGE1_GRAD_ACCUM_STEPS).backward()
            values = {"loss": loss.detach(), **metrics}
            if accumulated is None:
                accumulated = {
                    name: value / STAGE1_GRAD_ACCUM_STEPS
                    for name, value in values.items()
                }
            else:
                for name, value in values.items():
                    accumulated[name] += value / STAGE1_GRAD_ACCUM_STEPS

        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print(
            f"stage1 {step + 1:4d}/{total_steps} | "
            f"loss {accumulated['loss']:.4f} | "
            f"policy {accumulated['policy']:.4f} | "
            f"value {accumulated['value']:.4f} | "
            f"reward {accumulated['reward']:+.4f} | "
            f"factor_acc {accumulated['factor_acc']:.4f} | "
            f"tick_acc {accumulated['tick_acc']:.4f} | "
            f"entropy {accumulated['entropy']:.4f} | "
            f"norm {gradient_norm:.4f} | lr {learning_rate:.2e} | "
            f"dt {time.time() - start_time:.2f}s"
        )

    validation = evaluate_stage1(eval_model, val_loader)
    test = evaluate_stage1(eval_model, test_loader)
    print(
        f"stage1 final val | nll {validation['nll']:.4f} | "
        f"factor_acc {validation['factor_acc']:.4f} | "
        f"tick_acc {validation['tick_acc']:.4f}"
    )
    print(
        f"stage1 final test-long | nll {test['nll']:.4f} | "
        f"factor_acc {test['factor_acc']:.4f} | "
        f"tick_acc {test['tick_acc']:.4f}"
    )


def valid_action_logits(logits):
    """Mask PAD/START, which are sequence sentinels rather than actions."""
    minimum_tokens = (
        MOVE_STAY,
        MOVE_STAY,
        WRITE_NOOP,
        WRITE_NOOP,
        ROLE_WRITE_NOOP,
        ROLE_WRITE_NOOP,
    )
    masked = []
    for factor_logits, minimum in zip(logits, minimum_tokens):
        factor_logits = factor_logits.clone()
        factor_logits[..., :minimum] = torch.finfo(factor_logits.dtype).min
        masked.append(factor_logits)
    return tuple(masked)


def sample_factorized_action(logits):
    actions = []
    for factor_logits in logits:
        factor_logits = factor_logits[:, 0]
        factor_log_probs = F.log_softmax(factor_logits, dim=-1)
        probabilities = factor_log_probs.exp()
        action = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
        actions.append(action)
    return torch.stack(actions, dim=-1)


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
    return torch.stack(log_probs, dim=-1), torch.stack(entropies, dim=-1)


def reference_num_ticks(task, a, b):
    """Generate the Stage 1 trajectory only to measure its length."""
    if task == "add":
        return generate_add(a, b).num_ticks
    if task == "mul":
        return generate_mul(a, b).num_ticks
    raise ValueError(f"unknown task {task!r}")


@dataclass
class Episode:
    env: RoleTapeEnvironment
    teacher: RoleAwareTeacher
    budget: int
    budget_used: int = 0
    done: bool = False
    success: bool = False
    interventions: int = 0


@dataclass
class StoredTrajectory:
    reads: torch.Tensor
    previous_actions: torch.Tensor
    proposed_actions: torch.Tensor
    rewards: torch.Tensor

    @property
    def num_ticks(self):
        return self.rewards.size(0)


def initialize_episodes(items):
    episodes = []
    for task, a, b in items:
        teacher = RoleAwareTeacher(task, a, b)
        budget = math.ceil(
            reference_num_ticks(task, a, b) * STAGE2_BUDGET_FACTOR
        )
        episodes.append(
            Episode(
                env=RoleTapeEnvironment(task, a, b),
                teacher=teacher,
                budget=max(budget, 1),
            )
        )
    return episodes


def initialize_evaluation_episodes(items):
    episodes = []
    for task, a, b in items:
        budget = math.ceil(
            reference_num_ticks(task, a, b) * STAGE2_BUDGET_FACTOR
        )
        episodes.append(
            Episode(
                env=RoleTapeEnvironment(task, a, b),
                teacher=RoleAwareTeacher(task, a, b),
                budget=max(budget, 1),
            )
        )
    return episodes


def sample_stage2_items(rng, batch_size):
    items = []
    for _ in range(batch_size):
        task = rng.choice(TASKS)
        source = TRAIN_ADD if task == "add" else TRAIN_MUL
        a, b = rng.choice(source)
        items.append((task, a, b))
    return items


@torch.no_grad()
def collect_rollout_batch(model, items):
    episodes = initialize_episodes(items)
    buffers = [
        {
            "reads": [],
            "previous_actions": [],
            "proposed_actions": [],
            "rewards": [],
        }
        for _ in episodes
    ]
    states = None

    while any(not episode.done for episode in episodes):
        reads_rows = [episode.env.reads() for episode in episodes]
        previous_rows = [episode.env.prev_action for episode in episodes]
        reads = torch.tensor(
            reads_rows, dtype=torch.long, device=DEVICE
        ).unsqueeze(1)
        previous = torch.tensor(
            previous_rows, dtype=torch.long, device=DEVICE
        ).unsqueeze(1)
        logits, _, _, states = model(
            reads,
            previous,
            latent=None,
            states=states,
        )
        proposals = sample_factorized_action(
            valid_action_logits(logits)
        ).tolist()

        for index, episode in enumerate(episodes):
            if episode.done:
                continue
            proposal = tuple(proposals[index])
            buffer = buffers[index]
            buffer["reads"].append(reads_rows[index])
            # Input actions are what the environment actually executed.
            buffer["previous_actions"].append(previous_rows[index])
            # Output actions remain the policy proposals, even when rejected.
            buffer["proposed_actions"].append(proposal)

            decision = episode.teacher.step(episode.env, proposal)
            buffer["rewards"].append(decision.reward)
            episode.budget_used += decision.budget_cost
            episode.interventions += int(decision.intervened)
            episode.success = decision.success and not decision.intervened
            if decision.success or episode.budget_used >= episode.budget:
                episode.done = True

    trajectories = [
        StoredTrajectory(
            reads=torch.tensor(buffer["reads"], dtype=torch.long),
            previous_actions=torch.tensor(
                buffer["previous_actions"], dtype=torch.long
            ),
            proposed_actions=torch.tensor(
                buffer["proposed_actions"], dtype=torch.long
            ),
            rewards=torch.tensor(buffer["rewards"], dtype=torch.float32),
        )
        for buffer in buffers
    ]
    ticks = sum(trajectory.num_ticks for trajectory in trajectories)
    statistics = {
        "rollouts": len(episodes),
        "successes": sum(episode.success for episode in episodes),
        "ticks": ticks,
        "interventions": sum(episode.interventions for episode in episodes),
        "reward_sum": sum(
            float(trajectory.rewards.sum().item())
            for trajectory in trajectories
        ),
    }
    return trajectories, statistics


def collect_rollout_dataset(model, rng):
    model.eval()
    items = sample_stage2_items(rng, STAGE2_ROLLOUTS_PER_CYCLE)
    trajectories = []
    totals = {
        "rollouts": 0,
        "successes": 0,
        "ticks": 0,
        "interventions": 0,
        "reward_sum": 0.0,
    }
    for start in range(0, len(items), STAGE2_ROLLOUT_BATCH_SIZE):
        batch, statistics = collect_rollout_batch(
            model, items[start : start + STAGE2_ROLLOUT_BATCH_SIZE]
        )
        trajectories.extend(batch)
        for name, value in statistics.items():
            totals[name] += value

    ticks = max(totals["ticks"], 1)
    rollouts = max(totals["rollouts"], 1)
    return trajectories, {
        "success_rate": totals["successes"] / rollouts,
        "mean_ticks": totals["ticks"] / rollouts,
        "mean_reward": totals["reward_sum"] / ticks,
        "intervention_rate": totals["interventions"] / ticks,
    }


def collate_stored_trajectories(trajectories):
    batch_size = len(trajectories)
    max_ticks = max(trajectory.num_ticks for trajectory in trajectories)
    reads = torch.zeros(
        batch_size, max_ticks, config.num_inputs, dtype=torch.long
    )
    neutral = torch.tensor(NEUTRAL_ACTION, dtype=torch.long)
    previous_actions = neutral.view(1, 1, -1).repeat(
        batch_size, max_ticks, 1
    )
    proposed_actions = previous_actions.clone()
    rewards = torch.zeros(batch_size, max_ticks, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_ticks, dtype=torch.float32)

    for row, trajectory in enumerate(trajectories):
        ticks = trajectory.num_ticks
        reads[row, :ticks] = trajectory.reads
        previous_actions[row, :ticks] = trajectory.previous_actions
        proposed_actions[row, :ticks] = trajectory.proposed_actions
        rewards[row, :ticks] = trajectory.rewards
        mask[row, :ticks] = 1.0

    return tuple(
        tensor.to(DEVICE)
        for tensor in (
            reads,
            previous_actions,
            proposed_actions,
            rewards,
            mask,
        )
    )


def replay_trajectory_batch(model, optimizer, trajectories):
    (
        reads,
        previous_actions,
        proposed_actions,
        rewards,
        mask,
    ) = collate_stored_trajectories(trajectories)
    _, max_ticks = mask.shape
    valid_ticks = mask.sum().clamp_min(1.0)
    normalizer = valid_ticks * NUM_ACTION_FACTORS
    states = None
    policy_sum = mask.new_tensor(0.0)
    value_sum = mask.new_tensor(0.0)
    entropy_sum = mask.new_tensor(0.0)
    reward_sum = mask.new_tensor(0.0)

    model.train()
    optimizer.zero_grad()
    for start in range(0, max_ticks, config.block_size):
        end = min(start + config.block_size, max_ticks)
        logits, factor_values, _, states = model(
            reads[:, start:end].contiguous(),
            previous_actions[:, start:end].contiguous(),
            latent=None,
            states=states,
        )
        log_probs, entropies = fixed_action_statistics(
            valid_action_logits(logits),
            proposed_actions[:, start:end].contiguous(),
        )
        values = factor_values
        factor_rewards = (
            rewards[:, start:end]
            .to(values.dtype)
            .unsqueeze(-1)
            .expand_as(values)
        )
        factor_mask = (
            mask[:, start:end]
            .to(values.dtype)
            .unsqueeze(-1)
            .expand_as(values)
        )
        advantages = (factor_rewards - values).detach()
        policy_sum = policy_sum + (
            -log_probs * advantages * factor_mask
        ).sum()
        value_sum = value_sum + (
            (values - factor_rewards).square() * factor_mask
        ).sum()
        entropy_sum = entropy_sum + (entropies * factor_mask).sum()
        reward_sum = reward_sum + (
            rewards[:, start:end] * mask[:, start:end]
        ).sum()

    loss = (
        policy_sum
        + STAGE2_VALUE_COEF * value_sum
        - STAGE2_ENTROPY_COEF * entropy_sum
    ) / normalizer
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {
        "loss": loss.detach(),
        "policy": (policy_sum / normalizer).detach(),
        "value": (value_sum / normalizer).detach(),
        "entropy": (entropy_sum / normalizer).detach(),
        "reward": (reward_sum / valid_ticks).detach(),
        "gradient_norm": gradient_norm.detach(),
        "ticks": valid_ticks.detach(),
    }


def replay_stored_trajectories(model, optimizer, trajectories, rng):
    totals = None
    total_ticks = 0.0
    updates = 0
    indices = list(range(len(trajectories)))

    for _ in range(STAGE2_REPLAY_EPOCHS_PER_CYCLE):
        rng.shuffle(indices)
        for start in range(0, len(indices), STAGE2_REPLAY_BATCH_SIZE):
            batch = [
                trajectories[index]
                for index in indices[start : start + STAGE2_REPLAY_BATCH_SIZE]
            ]
            metrics = replay_trajectory_batch(model, optimizer, batch)
            ticks = float(metrics.pop("ticks").item())
            if totals is None:
                totals = {
                    name: float(value.item()) * ticks
                    for name, value in metrics.items()
                }
            else:
                for name, value in metrics.items():
                    totals[name] += float(value.item()) * ticks
            total_ticks += ticks
            updates += 1

    result = {
        name: total / max(total_ticks, 1.0)
        for name, total in totals.items()
    }
    result["updates"] = updates
    return result


@torch.no_grad()
def evaluate_stage2(model, items):
    model.eval()
    episodes = initialize_evaluation_episodes(items)
    states = None

    while any(not episode.done for episode in episodes):
        reads = torch.tensor(
            [episode.env.reads() for episode in episodes],
            dtype=torch.long,
            device=DEVICE,
        ).unsqueeze(1)
        previous = torch.tensor(
            [episode.env.prev_action for episode in episodes],
            dtype=torch.long,
            device=DEVICE,
        ).unsqueeze(1)
        logits, _, _, states = model(
            reads, previous, latent=None, states=states
        )
        logits = valid_action_logits(logits)
        actions = torch.stack(
            [factor[:, 0].argmax(dim=-1) for factor in logits],
            dim=-1,
        )
        proposed_actions = actions.tolist()
        for index, episode in enumerate(episodes):
            if episode.done:
                continue
            try:
                episode.env.apply(tuple(proposed_actions[index]))
            except InvalidAction:
                episode.done = True
                continue
            episode.budget_used += 1
            episode.success = episode.teacher.is_success(episode.env)
            if episode.success or episode.budget_used >= episode.budget:
                episode.done = True

    return {
        "success_rate": sum(episode.success for episode in episodes)
        / len(episodes),
        "mean_ticks": sum(episode.budget_used for episode in episodes)
        / len(episodes),
    }


def train_stage2(model, eval_model, optimizer):
    for group in optimizer.param_groups:
        group["lr"] = STAGE2_LR
    rng = random.Random(2026)
    validation_items = (
        [("add", a, b) for a, b in VAL_ADD[:32]]
        + [("mul", a, b) for a, b in VAL_MUL[:32]]
    )

    for cycle in range(STAGE2_TRAINING_CYCLES):
        if cycle % STAGE2_EVAL_INTERVAL_CYCLES == 0:
            validation = evaluate_stage2(eval_model, validation_items)
            print(
                f"stage2 {cycle:3d}/{STAGE2_TRAINING_CYCLES} | "
                f"val_success {validation['success_rate']:.4f} | "
                f"val_ticks {validation['mean_ticks']:.1f}"
            )

        start_time = time.time()
        trajectories, rollout_metrics = collect_rollout_dataset(model, rng)
        collection_time = time.time() - start_time
        replay_metrics = replay_stored_trajectories(
            model, optimizer, trajectories, rng
        )
        print(
            f"stage2 {cycle + 1:3d}/{STAGE2_TRAINING_CYCLES} | "
            f"loss {replay_metrics['loss']:.4f} | "
            f"policy {replay_metrics['policy']:.4f} | "
            f"value {replay_metrics['value']:.4f} | "
            f"entropy {replay_metrics['entropy']:.4f} | "
            f"reward {rollout_metrics['mean_reward']:+.4f} | "
            f"intervene {rollout_metrics['intervention_rate']:.4f} | "
            f"success {rollout_metrics['success_rate']:.4f} | "
            f"ticks {rollout_metrics['mean_ticks']:.1f} | "
            f"updates {replay_metrics['updates']} | "
            f"collect_dt {collection_time:.2f}s | "
            f"total_dt {time.time() - start_time:.2f}s"
        )

    test_items = (
        [("add", a, b) for a, b in TEST_LONG_ADD[:32]]
        + [("mul", a, b) for a, b in TEST_LONG_MUL[:32]]
    )
    test = evaluate_stage2(eval_model, test_items)
    print(
        f"stage2 final test-long | success {test['success_rate']:.4f} | "
        f"ticks {test['mean_ticks']:.1f}"
    )


def main():
    if config.num_inputs != 4 or config.num_outputs != NUM_ACTION_FACTORS:
        raise ValueError("role-channel training requires 4 inputs and 6 outputs")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    raw_model = GPT(config).to(DEVICE)
    model = torch.compile(raw_model) if COMPILE else raw_model
    parameters = sum(parameter.numel() for parameter in raw_model.parameters())
    print(f"Model: {parameters:,} parameters")
    print(f"Config: {config}")
    print(f"Input streams: {config.input_streams}")
    print(f"Output streams: {config.output_streams}")
    print(f"Compile: {COMPILE} | device: {DEVICE}")
    print("Stage 1: parallel dense teacher-state actor-critic")
    print(f"Stage 2 enabled: {RUN_STAGE2}")
    print("Stage 2 latent feedback: disabled")

    with open(log_file, "w"):
        pass
    stage1_optimizer = configure_optimizer(model, STAGE1_MAX_LR)
    train_stage1(model, raw_model, stage1_optimizer)
    stage1_checkpoint = os.path.join(
        log_dir, "tick_arith_role_stage1_mamba3_model.pt"
    )
    torch.save(
        {"model": raw_model.state_dict(), "config": config},
        stage1_checkpoint,
    )
    print(f"Saved Stage 1 model to {stage1_checkpoint}")

    if not RUN_STAGE2:
        print("Stage 2 disabled; stopping after Stage 1")
        return

    train_stage2(model, raw_model, stage1_optimizer)
    checkpoint = os.path.join(
        log_dir, "tick_arith_role_stage2_mamba3_model.pt"
    )
    torch.save({"model": raw_model.state_dict(), "config": config}, checkpoint)
    print(f"Saved model to {checkpoint}")


if __name__ == "__main__":
    main()
