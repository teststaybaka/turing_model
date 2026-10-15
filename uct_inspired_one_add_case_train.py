"""Frozen baseline: UCT-inspired learned search on one addition case.

Defaults reproduce the completed 40000-episode run from random weights:
16 rollouts, local-action credit, success +1, surprise x1, half-entropy
threshold, every-visit statistics, and no historical baseline or pretraining.
Configuration defaults below are the authoritative settings for this baseline.
Files are written directly under log/ with this script's name as their prefix.

Question: 11_01_____ (11 + 01). Submission checks only cells 6..10
(one-based): _100_. Every cell, including the operands, remains writable.

One search episode contains many fresh-tape rollouts. The model is frozen
throughout collection; a single REINFORCE update follows the entire episode.
Sparse per-(tape, action) n/W statistics persist between
episodes with lazy multiplicative decay. No action-policy table or PUCT prior.
Each rollout's binary outcome is backed up with fixed code. Its intrinsic
reward sums Beta posterior KL changes over each distinct state-action pair,
once per rollout by default. Optional backup variants erase closed tape loops
or count every repeated occurrence. Both use the same helpers as fixed MCTS.
All executed decisions remain in the REINFORCE batch and consume the horizon;
loop erasure affects statistics and surprise, not the sampled action history.
An optional success bonus is added once per correct submission, not per edge.
The surprise weight scales the training reward only; raw surprise and binary
statistics are preserved for comparisons.
The centered_predictive mode instead rewards observed outcome surprisal minus
its predicted expectation (Bernoulli entropy), using pre-backup smoothed counts.
surprise_threshold_scale optionally scales that entropy; below 1 the reward
is no longer zero in expectation under the predictive belief.
Original Beta-KL surprise remains a separate diagnostic in that mode.
With local_action credit, each executed action uses only its own signed
intrinsic signal plus its rollout's success bonus. Neither other actions'
intrinsic signals nor later rollout rewards are included in its policy weight.

Run with the repository Python: python uct_inspired_one_add_case_train.py
Use --output DIR for a separate run; existing baseline logs are not overwritten.
Checkpoints include optimizer, RNGs, and retained statistics for continuation.
"""

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


K = 10
SYMBOLS = 3  # 0, 1, blank
BLANK = 2
SUBMIT = K * SYMBOLS
NUM_ACTIONS = SUBMIT + 1
TAPE_STATES = SYMBOLS ** K
POWERS = SYMBOLS ** np.arange(K, dtype=np.int64)
INITIAL_TAPE = np.array([1, 1, BLANK, 0, 1, BLANK, BLANK, BLANK, BLANK, BLANK], dtype=np.int64)
EXPECTED_SUFFIX = np.array([BLANK, 1, 0, 0, BLANK], dtype=np.int64)
INPUT_SIZE = K * SYMBOLS + 2 * NUM_ACTIONS
STATISTICS_KEY = "tape_only"
BACKUP_RULE = "once_per_pair_per_rollout"
EVERY_VISIT_RULE = "every_pair_occurrence_per_rollout"


@dataclass
class Config:
    episodes: int = 40000
    rollouts: int = 16
    horizon: int = 64
    decay: float = 0.99
    hidden: int = 64
    learning_rate: float = 3e-4
    success_bonus: float = 1.0
    surprise_weight: float = 1.0
    surprise_mode: str = 'centered_predictive'
    surprise_threshold_scale: float = 0.5
    credit_assignment: str = 'local_action'
    baseline: str = "none"
    baseline_decay: float = 0.95
    seed: int = 20261011
    log_every: int = 1000
    checkpoint_every: int = 4000
    path_processing: str = 'none'
    backup_mode: str = 'every_visit'


def backup_every_visit(statistics, path, success):
    """Apply one count/outcome update per occurrence, including repeated pairs.

    Batch the marginal KL computations, using successive counts for each repeat.
    These repeated binary outcomes are correlated; surprise is a diagnostic.
    """
    multiplicities = Counter(path)
    counts, wins = [], []
    for (idx, action), repetitions in multiplicities.items():
        values = statistics.values(idx)
        counts.extend(float(values[0, action]) + step for step in range(repetitions))
        wins.extend(float(values[1, action]) + step * float(success) for step in range(repetitions))
    surprise = float(beta_surprise(np.asarray(counts), np.asarray(wins), success).sum())
    for (idx, action), repetitions in multiplicities.items():
        values = statistics.values(idx)
        values[0, action] += repetitions
        values[1, action] += repetitions * float(success)
    return surprise


def configured_backup_rule(config):
    return EVERY_VISIT_RULE if config.backup_mode == 'every_visit' else BACKUP_RULE


def predictive_surprise(counts, wins, success):
    """Outcome surprisal and its dynamic threshold, before an observation.

    p=(W+1)/(N+2). Threshold is E[-log P(outcome)]=Bernoulli entropy.
    Signed intrinsic reward is observed surprisal minus this threshold.
    Log-count arithmetic avoids rounding p to 0 or 1 for large counts.
    """
    n, w = np.asarray(counts, dtype=np.float64), np.asarray(wins, dtype=np.float64)
    if not np.isfinite(n).all() or not np.isfinite(w).all() or np.any(w < 0) or np.any(w > n):
        raise ValueError('predictive statistics require finite 0 <= W <= N')
    log_total = np.log(n + 2.0)
    log_success = np.log1p(w) - log_total
    log_failure = np.log1p(n - w) - log_total
    threshold = -(np.exp(log_success)*log_success + np.exp(log_failure)*log_failure)
    observed = -log_success if success else -log_failure
    return observed, threshold, observed - threshold


def path_predictive_surprise(statistics, path, success, every_visit, chronological=False):
    """Read-only reward calculation matching subsequent backup multiplicities.

    Repeated occurrences use successive marginal counts, just like every-visit
    Beta-KL backup. Their outcomes are correlated, not independent evidence.
    """
    counts, wins = [], []
    if chronological:
        if not every_visit:
            raise ValueError('chronological action credit requires every_visit')
        occurrences = Counter()
        for idx, action in path:
            offset = occurrences[idx, action]
            values = statistics.values(idx)
            counts.append(float(values[0, action]) + offset)
            wins.append(float(values[1, action]) + offset * float(success))
            occurrences[idx, action] += 1
    else:
        for (idx, action), repetitions in Counter(path).items():
            values = statistics.values(idx)
            for offset in range(repetitions if every_visit else 1):
                counts.append(float(values[0, action]) + offset)
                wins.append(float(values[1, action]) + offset * float(success))
    return predictive_surprise(counts, wins, success)


def erase_path_loops(path, states, submitted=False):
    """Erase closed tape-state detours; preserve a terminal SUBMIT edge.

    states contains the initial tape and the tape after every executed action.
    Reaching a retained state truncates the retained path back to that state.
    SUBMIT changes termination status even though it does not change the tape.
    """
    if len(states) != len(path) + 1:
        raise ValueError('expected one more state than action')
    retained, retained_states = [], [states[0]]
    positions = {states[0]: 0}
    for step, edge in enumerate(path):
        if submitted and step == len(path) - 1:
            retained.append(edge)
            break
        next_state = states[step + 1]
        if next_state in positions:
            position = positions[next_state]
            for removed in retained_states[position + 1:]:
                del positions[removed]
            del retained_states[position + 1:]
            del retained[position:]
        else:
            retained.append(edge)
            positions[next_state] = len(retained_states)
            retained_states.append(next_state)
    return retained


def answer_is_correct(tape):
    """Ignore input cells; require the separator, answer, and padding exactly."""
    return bool(np.array_equal(tape[5:], EXPECTED_SUFFIX))


def tape_text(tape):
    return "".join("_" if int(symbol) == BLANK else str(int(symbol)) for symbol in tape)


class Tape:
    def __init__(self):
        self.cells = INITIAL_TAPE.copy()
        self.code = int(self.cells @ POWERS)

    def step(self, action):
        """Return (submitted, success). A correct tape alone is not terminal."""
        if not 0 <= action < NUM_ACTIONS:
            raise ValueError("action out of range")
        if action == SUBMIT:
            return True, answer_is_correct(self.cells)
        position, symbol = divmod(action, SYMBOLS)
        self.code += int((symbol - self.cells[position]) * POWERS[position])
        self.cells[position] = symbol
        return False, False


class Statistics:
    """Block-allocated sparse rows; each row holds only action counts and sums.

    A tape shares one row across depths and rollouts. The estimates therefore
    pool outcomes from different remaining horizons. Episode age supports lazy
    exact decay. Search budgets are neither controller inputs nor row keys.
    """

    block_size = 4096

    def __init__(self, decay):
        self.decay = decay
        self.index = {}
        self.blocks = []
        self.ages = []

    def __len__(self):
        return len(self.index)

    def row(self, tape_code, episode):
        key = int(tape_code)
        if not 0 <= key < TAPE_STATES:
            raise ValueError("statistics key must be a valid tape configuration")
        idx = self.index.get(key)
        if idx is None:
            idx = len(self.index)
            self.index[key] = idx
            if idx % self.block_size == 0:
                self.blocks.append(np.zeros((self.block_size, 2, NUM_ACTIONS), dtype=np.float32))
                self.ages.append(np.zeros(self.block_size, dtype=np.int64))
            block, offset = divmod(idx, self.block_size)
            self.ages[block][offset] = episode
        block, offset = divmod(idx, self.block_size)
        age = int(self.ages[block][offset])
        if episode < age:
            raise ValueError("statistics cannot move backwards in episode time")
        values = self.blocks[block][offset]
        if episode != age:
            values *= self.decay ** (episode - age)
            self.ages[block][offset] = episode
        return idx, values

    def values(self, idx):
        block, offset = divmod(idx, self.block_size)
        return self.blocks[block][offset]

    def backup(self, path, success):
        """Sum marginal surprises; all edges receive the same binary outcome.

        Update each distinct (tape, action) pair once per rollout. Repeated
        occurrences share one outcome, not independent evidence. Different
        actions from one tape still receive separate updates. The sum across
        pairs is a reward definition, not a claim of independent information.
        """
        unique_pairs = list(dict.fromkeys(path))
        counts = np.array([self.values(idx)[0, action] for idx, action in unique_pairs], dtype=np.float64)
        wins = np.array([self.values(idx)[1, action] for idx, action in unique_pairs], dtype=np.float64)
        surprises = beta_surprise(counts, wins, success)
        for idx, action in unique_pairs:
            values = self.values(idx)
            values[0, action] += 1.0
            values[1, action] += float(success)
        return float(surprises.sum())

    def snapshot(self):
        count = len(self)
        return {
            "key_type": STATISTICS_KEY,
            "backup_rule": BACKUP_RULE,
            "keys": np.fromiter(self.index.keys(), dtype=np.int64, count=count),
            "values": np.concatenate(self.blocks, axis=0)[:count] if count else np.empty((0, 2, NUM_ACTIONS), dtype=np.float32),
            "ages": np.concatenate(self.ages)[:count] if count else np.empty(0, dtype=np.int64),
        }

    @classmethod
    def restore(cls, decay, snapshot):
        if snapshot.get("key_type") != STATISTICS_KEY or snapshot.get("backup_rule") != BACKUP_RULE:
            raise ValueError("incompatible statistics: start a fresh tape-only run; legacy horizon-keyed checkpoints cannot be resumed")
        result = cls(decay)
        keys = snapshot["keys"]
        if np.any(keys < 0) or np.any(keys >= TAPE_STATES) or len(np.unique(keys)) != len(keys):
            raise ValueError("invalid or duplicate tape-only statistics keys")
        result.index = {int(key): idx for idx, key in enumerate(keys)}
        for start in range(0, len(keys), cls.block_size):
            length = min(cls.block_size, len(keys) - start)
            values = np.zeros((cls.block_size, 2, NUM_ACTIONS), dtype=np.float32)
            ages = np.zeros(cls.block_size, dtype=np.int64)
            values[:length] = snapshot["values"][start:start + length]
            ages[:length] = snapshot["ages"][start:start + length]
            result.blocks.append(values)
            result.ages.append(ages)
        return result


def beta_surprise(counts, wins, success):
    """KL(Beta after one Bernoulli observation || Beta before).

    a=1+W, b=1+n-W; closed-form simplification avoids subtracting log-beta
    functions. Float64 avoids cancellation for repeatedly visited edges.
    """
    n = torch.as_tensor(counts, dtype=torch.float64)
    w = torch.as_tensor(wins, dtype=torch.float64)
    if torch.any(w < 0) or torch.any(n < w):
        raise ValueError("Beta statistics require 0 <= W <= n")
    total = n + 2.0
    observed = 1.0 + (w if success else n - w)
    kl = torch.log(total / observed) + torch.special.digamma(observed + 1.0) - torch.special.digamma(total + 1.0)
    return kl.clamp_min(0.0).numpy()


class SearchController(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, NUM_ACTIONS),
        )
        # Begin close to uniform without any arithmetic supervision.
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


def features(tape, values):
    x = np.zeros(INPUT_SIZE, dtype=np.float32)
    x[np.arange(K) * SYMBOLS + tape] = 1.0
    # No Q or separate policy is fed in; retain both n and W explicitly.
    x[K * SYMBOLS:K * SYMBOLS + 2 * NUM_ACTIONS] = np.log1p(values).reshape(-1) / 8.0
    return x


def undiscounted_returns(rewards):
    return np.cumsum(np.asarray(rewards, dtype=np.float64)[::-1])[::-1].copy()


class PositionBaseline:
    """Historical return average per rollout position; never a model input.

    Compute advantages before updating. The first policy update uses zero;
    its observed returns initialize the averages for the next episode.
    """
    def __init__(self, rollouts, decay):
        self.values = np.zeros(rollouts, dtype=np.float64)
        self.decay = decay
        self.updates = 0

    def advantages(self, returns):
        returns = np.asarray(returns, dtype=np.float64)
        if returns.shape != self.values.shape or not np.isfinite(returns).all():
            raise ValueError('invalid per-rollout returns')
        return returns - self.values

    def update(self, returns):
        self.advantages(returns)  # Validate before changing stored state.
        if self.updates == 0:
            self.values[:] = returns
        else:
            self.values *= self.decay
            self.values += (1 - self.decay) * np.asarray(returns)
        self.updates += 1

    def snapshot(self):
        return {'values': self.values.copy(), 'decay': self.decay, 'updates': self.updates}

    @classmethod
    def restore(cls, snapshot, rollouts, decay):
        if snapshot['decay'] != decay or snapshot['updates'] < 0:
            raise ValueError('incompatible baseline checkpoint')
        result = cls(rollouts, decay)
        result.advantages(snapshot['values'])
        result.values[:] = snapshot['values']
        result.updates = int(snapshot['updates'])
        return result


def reinforce_loss(logits, actions, returns, rollouts):
    """Negative log-probability weighted by detached returns or advantages."""
    selected = F.log_softmax(logits, dim=-1).gather(1, actions[:, None]).squeeze(1)
    return -(selected * returns.detach()).sum() / rollouts


def policy_weights(batch, baseline):
    """Subtract prior-episode position means without replacing local signals."""
    if baseline is None:
        return batch['returns']
    offsets = np.repeat(baseline.values, batch['rollout_lengths'].numpy())
    return batch['returns'] - torch.tensor(offsets, dtype=batch['returns'].dtype)


def collect_episode(model, statistics, config, episode, rng, uniform=False):
    if config.credit_assignment not in ('rollout_return', 'local_action'):
        raise ValueError('unknown credit_assignment')
    local_credit = config.credit_assignment == 'local_action'
    if local_credit and (config.surprise_mode != 'centered_predictive' or
                         config.backup_mode != 'every_visit' or config.path_processing != 'none'):
        raise ValueError('local_action requires centered_predictive, every_visit and no path processing')
    xs, actions, lengths, rewards, old_logps = [], [], [], [], []
    action_credits, rollout_credit_means, action_success_credits = [], [], []
    surprises = []
    intrinsic_rewards, outcome_surprisals, thresholds = [], [], []
    predictive_entropies = []
    positive_intrinsic_edges = negative_intrinsic_edges = 0
    successes = submissions = timeouts = answer_reached = unique_backups = 0
    repeated = erased = retained_actions = backup_updates = 0
    successful_paths = []
    with torch.inference_mode():
        for rollout in range(config.rollouts):
            tape = Tape()
            path = []
            states = [tape.code]
            rollout_actions = []
            success = submitted = reached = False
            for _ in range(config.horizon):
                idx, values = statistics.row(tape.code, episode)
                x = features(tape.cells, values)
                if uniform:
                    probabilities = np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS)
                else:
                    probabilities = torch.softmax(model(torch.from_numpy(x)), dim=-1).numpy()
                action = min(int(np.searchsorted(np.cumsum(probabilities), rng.random())), NUM_ACTIONS - 1)
                xs.append(x)
                actions.append(action)
                old_logps.append(math.log(float(probabilities[action])))
                path.append((idx, action))
                rollout_actions.append(action)
                submitted, success = tape.step(action)
                states.append(tape.code)
                reached = reached or answer_is_correct(tape.cells)
                if submitted:
                    break
            backup_path = (erase_path_loops(path, states, submitted)
                           if config.path_processing == 'erase_loops' else path)
            if config.surprise_mode == 'centered_predictive':
                observed, entropy, _ = path_predictive_surprise(
                    statistics, backup_path, success, config.backup_mode == 'every_visit',
                    chronological=local_credit)
                threshold = config.surprise_threshold_scale * entropy
                centered = observed - threshold
                predictive_entropies.append(float(entropy.sum()))
                outcome_surprisals.append(float(observed.sum()))
                thresholds.append(float(threshold.sum()))
                intrinsic = float(centered.sum())
                positive_intrinsic_edges += int((centered > 1e-12).sum())
                negative_intrinsic_edges += int((centered < -1e-12).sum())
                if local_credit:
                    credit = config.surprise_weight * centered + config.success_bonus * int(success)
                    assert len(credit) == len(path)
                    action_credits.extend(credit.tolist())
                    rollout_credit_means.append(float(credit.mean()))
                    action_success_credits.extend([config.success_bonus * int(success)] * len(path))
            if config.backup_mode == 'every_visit':
                surprise = backup_every_visit(statistics, backup_path, success)
                backup_updates += len(backup_path)
            else:
                surprise = statistics.backup(backup_path, success)
                backup_updates += len(set(backup_path))
            surprises.append(surprise)
            if config.surprise_mode == 'beta_kl':
                intrinsic = surprise
            intrinsic_rewards.append(intrinsic)
            rewards.append(config.surprise_weight * intrinsic + config.success_bonus * int(success))
            unique_backups += len(set(backup_path))
            repeated += len(path) - len(set(path))
            erased += len(path) - len(backup_path)
            retained_actions += len(backup_path)
            lengths.append(len(path))
            successes += int(success)
            submissions += int(submitted)
            timeouts += int(not submitted)
            answer_reached += int(reached)
            if success:
                successful_paths.append({"rollout": rollout, "actions": rollout_actions,
                                         "backup_actions": [action for _, action in backup_path],
                                         "tape": tape_text(tape.cells)})
    # Local-mode position EMA targets the mean action credit in that rollout,
    # using prior episodes only. It never broadcasts that mean as action credit.
    reward_returns = np.asarray(rollout_credit_means) if local_credit else undiscounted_returns(rewards)
    batch = {
        "features": torch.from_numpy(np.stack(xs)),
        "actions": torch.tensor(actions, dtype=torch.long),
        "returns": torch.tensor(action_credits if local_credit else np.repeat(reward_returns, lengths), dtype=torch.float32),
        "rollout_returns": torch.tensor(reward_returns, dtype=torch.float64),
        "rollout_lengths": torch.tensor(lengths, dtype=torch.long),
        "old_logps": torch.tensor(old_logps, dtype=torch.float32),
    }
    metrics = {
        "episode": episode + 1, "transitions": len(actions),
        "successes": successes, "success_rate": successes / config.rollouts,
        "submissions": submissions, "timeouts": timeouts,
        "timeout_rate": timeouts / config.rollouts,
        "answer_reached_rate": answer_reached / config.rollouts,
        "mean_length": float(np.mean(lengths)),
        "mean_rollout_surprise": float(np.mean(surprises)),
        "surprise_per_action": float(np.sum(surprises) / len(actions)),
        "mean_rollout_weighted_surprise": config.surprise_weight * float(np.mean(surprises)),
        "mean_rollout_intrinsic_reward": float(np.mean(intrinsic_rewards)),
        "mean_rollout_weighted_intrinsic_reward": config.surprise_weight * float(np.mean(intrinsic_rewards)),
        "mean_rollout_reward": float(np.mean(rewards)),
        "mean_rollout_success_bonus": config.success_bonus * successes / config.rollouts,
        "unique_backups": unique_backups,
        "repeat_pair_visits": repeated,
        "backup_updates": backup_updates,
        "loop_erased_actions": erased, "retained_path_actions": retained_actions,
        "statistics_rows": len(statistics),
    }
    if config.surprise_mode == 'centered_predictive':
        metrics.update(mean_rollout_outcome_surprisal=float(np.mean(outcome_surprisals)),
                       mean_rollout_predictive_entropy=float(np.mean(predictive_entropies)),
                       mean_rollout_surprise_threshold=float(np.mean(thresholds)),
                       positive_intrinsic_edges=positive_intrinsic_edges,
                       negative_intrinsic_edges=negative_intrinsic_edges)
    if local_credit:
        metrics.update(mean_action_credit=float(np.mean(action_credits)),
                       mean_action_intrinsic_credit=config.surprise_weight*float(np.sum(intrinsic_rewards))/len(actions),
                       mean_action_success_credit=float(np.mean(action_success_credits)),
                       mean_rollout_action_credit=float(np.mean(rollout_credit_means)))
    return batch, metrics, successful_paths


def replace_checkpoint(temporary, path):
    """Retry transient Windows file locks without discarding a complete save."""
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(0.25 * 2 ** attempt, 2.0))


def save_checkpoint(path, model, optimizer, statistics, config, episode, rng, baseline=None, frozen=False):
    snapshot = statistics.snapshot()
    snapshot['backup_rule'] = configured_backup_rule(config)
    payload = {
        "config": asdict(config), "completed_episodes": episode,
        "input_size": INPUT_SIZE,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "statistics": snapshot, "numpy_rng": rng.bit_generator.state,
        "torch_rng": torch.get_rng_state(),
        "baseline": baseline.snapshot() if baseline is not None else None,
        "frozen": frozen,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary, pickle_protocol=4)
    replace_checkpoint(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in asdict(Config()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    parser.add_argument("--output", type=Path, default=Path("log"),
                        help="Directory for script-prefixed logs and model files")
    parser.add_argument("--resume", type=Path, help="Load a trusted checkpoint produced by this script")
    parser.add_argument("--uct-initialize", type=Path, help="Initialize weights from a statistics-only UCT imitation checkpoint")
    parser.add_argument("--uniform", action="store_true", help="Fixed uniform controller diagnostic; no training")
    parser.add_argument('--frozen', action='store_true', help='Evaluate initialized neural weights without optimizer or baseline updates; search statistics still update')
    args = parser.parse_args()
    if args.frozen and (args.uniform or not (args.uct_initialize or args.resume)):
        parser.error('--frozen requires --uct-initialize or --resume and cannot use --uniform')
    if args.uct_initialize and (args.resume or args.uniform):
        parser.error("--uct-initialize requires a fresh learned run")
    config = Config(**{name: getattr(args, name) for name in asdict(Config())})
    if min(config.episodes, config.rollouts, config.horizon, config.hidden, config.log_every, config.checkpoint_every) < 1:
        parser.error("episode counts, horizon, hidden size, and intervals must be positive")
    if not 0 < config.decay <= 1 or config.learning_rate <= 0:
        parser.error("require 0 < decay <= 1 and learning_rate > 0")
    if not math.isfinite(config.success_bonus) or config.success_bonus < 0:
        parser.error("success_bonus must be finite and nonnegative")
    if not math.isfinite(config.surprise_weight) or config.surprise_weight < 0:
        parser.error('surprise_weight must be finite and nonnegative')
    if config.surprise_mode not in ('beta_kl', 'centered_predictive'):
        parser.error('surprise-mode must be beta_kl or centered_predictive')
    if not math.isfinite(config.surprise_threshold_scale) or config.surprise_threshold_scale < 0:
        parser.error('surprise_threshold_scale must be finite and nonnegative')
    if config.surprise_mode != 'centered_predictive' and config.surprise_threshold_scale != 1:
        parser.error('nondefault surprise_threshold_scale requires centered_predictive')
    if config.credit_assignment not in ('rollout_return', 'local_action'):
        parser.error('credit_assignment must be rollout_return or local_action')
    if config.credit_assignment == 'local_action' and (config.surprise_mode != 'centered_predictive' or
            config.backup_mode != 'every_visit' or config.path_processing != 'none'):
        parser.error('local_action requires centered_predictive, every_visit and path_processing none')
    if config.baseline not in ('none', 'position_ema') or not 0 <= config.baseline_decay < 1:
        parser.error('baseline must be none or position_ema; require 0 <= baseline_decay < 1')
    if config.path_processing not in ('none', 'erase_loops'):
        parser.error('path-processing must be none or erase_loops')
    if config.backup_mode not in ('distinct_pairs', 'every_visit'):
        parser.error('backup-mode must be distinct_pairs or every_visit')
    if config.backup_mode == 'every_visit' and config.path_processing != 'none':
        parser.error('every_visit requires the complete path (path-processing none)')
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = SearchController(config.hidden)
    if args.uct_initialize:
        initialization = torch.load(args.uct_initialize, map_location="cpu", weights_only=False)
        if initialization.get("teacher") != "uct_statistics_only":
            parser.error("expected a UCT statistics-imitation checkpoint")
        model.load_state_dict(initialization["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    statistics = Statistics(config.decay)
    baseline = PositionBaseline(config.rollouts, config.baseline_decay) if config.baseline == 'position_ema' else None
    start_episode = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get('frozen', False) != args.frozen:
            parser.error('resume requires matching frozen mode; do not mix evaluation with training')
        if checkpoint['config'].get('credit_assignment', 'rollout_return') != config.credit_assignment:
            parser.error('resume requires matching credit_assignment; baseline semantics also differ')
        if checkpoint["model"]["net.0.weight"].shape[1] != INPUT_SIZE:
            parser.error("checkpoint uses incompatible controller inputs; start a fresh run with tape and statistics only")
        for field, default in (('path_processing', 'none'), ('backup_mode', 'distinct_pairs')):
            if checkpoint['config'].get(field, default) != getattr(config, field):
                parser.error(f'resume requires matching {field}')
        if checkpoint['statistics'].get('backup_rule') != configured_backup_rule(config):
            parser.error('checkpoint statistics backup rule differs from config')
        try:
            storage = {**checkpoint['statistics'], 'backup_rule': BACKUP_RULE}
            statistics = Statistics.restore(config.decay, storage)
        except ValueError as error:
            parser.error(str(error))
        for field in ("rollouts", "horizon", "hidden", "decay", "learning_rate"):
            if checkpoint["config"][field] != getattr(config, field):
                parser.error(f"resume requires matching {field}")
        if checkpoint["config"].get("success_bonus", 0.0) != config.success_bonus:
            parser.error("resume requires matching success_bonus; use a fresh run to change the objective")
        if checkpoint['config'].get('surprise_weight', 1.0) != config.surprise_weight:
            parser.error('resume requires matching surprise_weight; use a fresh run to change the objective')
        if checkpoint['config'].get('surprise_mode', 'beta_kl') != config.surprise_mode:
            parser.error('resume requires matching surprise_mode; use a fresh run to change the objective')
        if checkpoint['config'].get('surprise_threshold_scale', 1.0) != config.surprise_threshold_scale:
            parser.error('resume requires matching surprise_threshold_scale; use a fresh run to change the objective')
        if checkpoint['config'].get('baseline', 'none') != config.baseline:
            parser.error('resume requires matching baseline')
        if baseline is not None:
            if checkpoint.get('baseline') is None:
                parser.error('missing baseline checkpoint state')
            try:
                baseline = PositionBaseline.restore(checkpoint['baseline'], config.rollouts, config.baseline_decay)
            except ValueError as error:
                parser.error(str(error))
            expected_updates = 0 if args.frozen else checkpoint['completed_episodes']
            if baseline.updates != expected_updates and not args.uniform:
                parser.error('baseline update count does not match completed episodes')
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        rng.bit_generator.state = checkpoint["numpy_rng"]
        torch.set_rng_state(checkpoint["torch_rng"])
        start_episode = checkpoint["completed_episodes"]
    if args.frozen:
        if optimizer.state:
            parser.error('frozen evaluation must not contain optimizer updates')
        model.eval()
        model.requires_grad_(False)
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = Path(__file__).stem
    metrics_path = args.output / f"{prefix}.metrics.jsonl"
    if metrics_path.exists() and not args.resume:
        parser.error("output already contains metrics; choose a new output directory or use --resume")
    run_config = {**asdict(config), "uniform": args.uniform, "device": "cpu", "parameters": sum(p.numel() for p in model.parameters()),
                  "frozen": args.frozen,
                  "intrinsic_reward_definition": (('outcome_surprisal_minus_predictive_entropy' if config.surprise_threshold_scale == 1 else 'outcome_surprisal_minus_scaled_predictive_entropy') if config.surprise_mode == 'centered_predictive' else 'beta_posterior_kl'),
                  "raw_surprise_metric": 'beta_posterior_kl_diagnostic',
                  "statistics_key": STATISTICS_KEY, "backup_rule": configured_backup_rule(config),
                  "backup_path": config.path_processing,
                  "policy_credit": "all_executed_actions",
                  "policy_credit_signal": ('own_intrinsic_plus_own_rollout_success' if config.credit_assignment == 'local_action' else 'current_and_later_rollout_rewards'),
                  "baseline_target": ('mean_action_credit_per_rollout_position' if config.credit_assignment == 'local_action' else 'future_rollout_return_per_position'),
                  "rollout_reward_role": ('summed_intrinsic_plus_once_success_diagnostic' if config.credit_assignment == 'local_action' else 'training_reward_before_future_rollout_returns'),
                  "input_size": INPUT_SIZE, "controller_inputs": "tape_and_statistics_only"}
    if args.uct_initialize:
        run_config["uct_initialized_from"] = str(args.uct_initialize.resolve())
    print(json.dumps(run_config), flush=True)
    print('CPU inference: frozen weights; statistics update after each rollout.' if args.frozen else
          "CPU inference: tiny sequential decisions; one optimizer update per complete search episode.", flush=True)
    started = time.perf_counter()
    with metrics_path.open("a", encoding="utf-8", buffering=1) as log:
        for episode in range(start_episode, config.episodes):
            batch, metrics, successes = collect_episode(model, statistics, config, episode, rng, uniform=args.uniform)
            if not args.uniform and not args.frozen:
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch["features"])
                weights = batch['returns']
                if baseline is not None:
                    advantages = baseline.advantages(batch['rollout_returns'].numpy())
                    # Preserve legacy float64 subtraction before casting; local
                    # mode must retain distinct per-action credit values.
                    weights = (policy_weights(batch, baseline) if config.credit_assignment == 'local_action' else
                               torch.tensor(np.repeat(advantages, batch['rollout_lengths'].numpy()), dtype=torch.float32))
                    metrics.update(baseline_mean=float(baseline.values.mean()),
                                   mean_rollout_advantage=float(advantages.mean()),
                                   negative_advantage_fraction=float((advantages < 0).mean()))
                if config.credit_assignment == 'local_action':
                    metrics.update(mean_action_advantage=float(weights.mean()),
                                   negative_action_advantage_fraction=float((weights < 0).float().mean()))
                loss = reinforce_loss(logits, batch["actions"], weights, config.rollouts)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite loss")
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                if baseline is not None:
                    baseline.update(batch['rollout_returns'].numpy())
                    metrics['baseline_updates'] = baseline.updates
                with torch.no_grad():
                    log_probs = F.log_softmax(logits, dim=-1)
                    metrics["entropy"] = float(-(log_probs.exp() * log_probs).sum(-1).mean())
                metrics.update(loss=float(loss.detach()), gradient_norm=float(gradient_norm))
            elif args.frozen:
                with torch.inference_mode():
                    log_probs = F.log_softmax(model(batch['features']), dim=-1)
                    metrics['entropy'] = float(-(log_probs.exp() * log_probs).sum(-1).mean())
                metrics.update(optimizer_updates=0, baseline_updates=0)
            metrics["elapsed_seconds"] = time.perf_counter() - started
            log.write(json.dumps(metrics) + "\n")
            if successes:
                with (args.output / f"{prefix}.successes.jsonl").open("a", encoding="utf-8") as success_file:
                    for example in successes:
                        success_file.write(json.dumps({"episode": episode + 1, **example}) + "\n")
            if episode == start_episode or (episode + 1) % config.log_every == 0:
                print(f"episode {episode + 1}/{config.episodes} | success {metrics['success_rate']:.3f} | "
                      f"length {metrics['mean_length']:.1f} | timeout {metrics['timeout_rate']:.3f} | "
                      f"surprise/rollout {metrics['mean_rollout_surprise']:.3f} | "
                      f"surprise/action {metrics['surprise_per_action']:.4f} | "
                      f"states {len(statistics)} | seconds {metrics['elapsed_seconds']:.1f}", flush=True)
            if (episode + 1) % config.checkpoint_every == 0 or episode + 1 == config.episodes:
                save_checkpoint(args.output / f"{prefix}.checkpoint.pt", model, optimizer, statistics, config, episode + 1, rng, baseline, frozen=args.frozen)
    if config.episodes > start_episode:
        model_path = args.output / f"{prefix}.model.pt"
        temporary = model_path.with_suffix('.tmp')
        torch.save({'model': model.state_dict(), 'config': asdict(config),
                    'input_size': INPUT_SIZE, 'completed_episodes': config.episodes}, temporary)
        replace_checkpoint(temporary, model_path)
    print(f"Finished. Metrics and resumable checkpoint: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
