"""Reproducible UCT baseline on all 16 ordered additions with operands 0..3.

Run `python uct_16_questions_baseline.py` with no arguments to reproduce the
final seed-20261011 experiment: 36,000 shuffled passes, 576,000 updates,
16 rollouts/update, horizon 64, decay .99, and 10x signed W/N with c=1.
Optional --seed 20261012 or --seed 20261013 reproduces the other two seeds.
Results go directly to log/ with prefix uct_16_questions_baseline_seed<seed>.

The ten-cell ternary tape starts [a_hi,a_lo,blank,b_hi,b_lo,blank,blank,
blank,blank,blank]. Operand cells 0,1,3,4 are protected. SUBMIT checks that
cells 5..9 equal [blank,sum_hi,sum_mid,sum_lo,blank]; cell 2 is writable.
There are 30 writes plus SUBMIT. Writable cells start blank every rollout.

Same Tape, question schedule, sampling, horizon, persistent tape-only rows,
own-question decay and after-rollout every-occurrence accounting as the neural
experiment. Only this UCT driver uses signed W; neural statistics are unchanged.
N += 1 per executed occurrence. W += 1 per occurrence in a successful rollout;
otherwise W += 0, except the terminal input-violation action gets W -= 1.
No penalty is propagated to preceding actions. No surprise or Beta calculation
is applied to signed W, whose valid range is -N <= W <= N.

At EVERY action, choose uniformly among unvisited actions; otherwise choose
uniformly among maxima of scale*(W/N) + c*sqrt(log(max(1,sum(N)))/N),
with scale=10 and c=1 by default. The scale multiplies the entire signed mean
during selection, including penalties; stored W retains +1/-1 increments.
No expansion, default-rollout phase, extra penalty table, masks, loop erasure,
within-rollout visits, network or optimizer. Rows still pool remaining horizons.
Only aggregate logs and a rolling signed-statistics/RNG checkpoint are saved.
Existing files for the same seed are never overwritten. No automatic resume
or monitoring.
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



K, SYMBOLS, BLANK = 10, 3, 2
SUBMIT = K*SYMBOLS
NUM_ACTIONS = SUBMIT+1
TAPE_STATES = SYMBOLS**K
POWERS = SYMBOLS**np.arange(K, dtype=np.int64)
PROTECTED_CELLS = (0, 1, 3, 4)


def question_operands(question_id):
    if not 0 <= question_id < 16:
        raise ValueError('question_id must be 0..15')
    return divmod(question_id, 4)

def question_schedule(config):
    rng = np.random.default_rng(config.seed+1)
    return [int(q) for _ in range(config.passes) for q in rng.permutation(16)]

def statistics_episode(config, episode):
    """Zero-based own-question age; each shuffled pass visits every question once."""
    local = episode % config.episodes_per_question
    if config.statistics_scope == 'question':
        return (episode // (16*config.episodes_per_question))*config.episodes_per_question + local
    return local

class Tape:
    def __init__(self, question_id):
        left, right = question_operands(question_id)
        self.cells = np.full(K, BLANK, dtype=np.int64)
        self.cells[[0, 1, 3, 4]] = [left//2, left%2, right//2, right%2]
        self.code = int(self.cells @ POWERS)
        total = left+right
        self.expected_suffix = np.array([BLANK, total//4, (total//2)%2, total%2, BLANK])

    def step(self, action):
        if not 0 <= action < NUM_ACTIONS:
            raise ValueError('action out of range')
        if action == SUBMIT:
            return True, bool(np.array_equal(self.cells[5:], self.expected_suffix)), False
        position, symbol = divmod(action, SYMBOLS)
        if position in PROTECTED_CELLS and symbol != self.cells[position]:
            return False, False, True
        self.code += int((symbol-self.cells[position])*POWERS[position])
        self.cells[position] = symbol
        return False, False, False


def atomic_save(payload, path):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(0.25*2**attempt, 2.0))


@dataclass
class Config:
    episodes_per_question: int = 1
    passes: int = 36000
    rollouts: int = 16
    horizon: int = 64
    success_bonus: float = 1.0
    input_change_penalty: float = -1.0
    decay: float = 0.99
    statistics_scope: str = 'question'
    exploration: float = 1.0
    success_value_scale: float = 10.0
    seed: int = 20261011
    log_every: int = 1000
    checkpoint_every: int = 1000

    @property
    def episodes(self):
        return 16*self.passes*self.episodes_per_question


SIGNED_BACKUP_RULE = 'every_pair_occurrence_success_plus1_offending_violation_minus1'


class Statistics:
    """Sparse tape-keyed signed counts, lazily decayed on own-question updates."""

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

    def backup(self, path, success, invalid):
        if not path or (success and invalid):
            raise ValueError('expected a nonempty rollout with exclusive terminal outcomes')
        for (idx, action), count in Counter(path).items():
            values = self.values(idx)
            values[0, action] += count
            values[1, action] += count*int(success)
        if invalid:
            idx, action = path[-1]
            self.values(idx)[1, action] -= 1.0

    def snapshot(self):
        count = len(self)
        return {
            "value_semantics": "successful_action_occurrences_minus_offending_input_violations",
            "key_type": 'tape_only',
            "backup_rule": SIGNED_BACKUP_RULE,
            "keys": np.fromiter(self.index.keys(), dtype=np.int64, count=count),
            "values": np.concatenate(self.blocks, axis=0)[:count] if count else np.empty((0, 2, NUM_ACTIONS), dtype=np.float32),
            "ages": np.concatenate(self.ages)[:count] if count else np.empty(0, dtype=np.int64),
        }


def uct_probabilities(values, exploration=1.0, success_value_scale=10.0):
    n, w = values.astype(np.float64, copy=False)
    candidates = np.flatnonzero(n == 0)
    if not len(candidates):
        scores = success_value_scale*(w/n) + exploration*np.sqrt(math.log(max(1.0, float(n.sum())))/n)
        candidates = np.flatnonzero(np.isclose(scores, scores.max(), rtol=1e-10, atol=1e-12))
    probabilities = np.zeros(len(n), dtype=np.float64)
    probabilities[candidates] = 1.0/len(candidates)
    return probabilities


class UCTPolicy:
    def __init__(self, exploration=1.0, success_value_scale=10.0):
        self.exploration = exploration
        self.success_value_scale = success_value_scale
        self.entropy_sum = 0.0
        self.decisions = 0

    def __call__(self, features, values):
        probabilities = uct_probabilities(values, self.exploration, self.success_value_scale)
        self.entropy_sum += math.log(np.count_nonzero(probabilities))
        self.decisions += 1
        return probabilities


def collect_episode(statistics, config, episode, rng, coverage, question_id, block_coverage):
    policy = UCTPolicy(config.exploration, config.success_value_scale)
    before = len(statistics)
    old_coverage, old_block_coverage = len(coverage), len(block_coverage)
    transitions = successes = submissions = invalids = timeouts = successful_actions = repeats = 0
    for _ in range(config.rollouts):
        tape = Tape(question_id)
        path = []
        seen_pairs = set()
        submitted = success = invalid = False
        for _ in range(config.horizon):
            idx, values = statistics.row(tape.code, statistics_episode(config, episode))
            probabilities = policy(None, values)
            action = min(int(np.searchsorted(np.cumsum(probabilities), rng.random())), NUM_ACTIONS-1)
            coverage.add(tape.code)
            block_coverage.add(tape.code)
            pair = (tape.code, action)
            repeats += int(pair in seen_pairs)
            seen_pairs.add(pair)
            path.append((idx, action))
            submitted, success, invalid = tape.step(action)
            if submitted or invalid:
                break
        statistics.backup(path, success, invalid)
        transitions += len(path)
        successful_actions += len(path)*int(success)
        successes += int(success)
        submissions += int(submitted)
        invalids += int(invalid)
        timeouts += int(not submitted and not invalid)
    assert policy.decisions == transitions
    left, right = question_operands(question_id)
    return dict(episode=episode+1, question_id=question_id, operand_left=left, operand_right=right,
                question_block=episode//config.episodes_per_question+1,
                pass_index=episode//(16*config.episodes_per_question)+1,
                episode_in_block=episode%config.episodes_per_question+1,
                statistics_episode=statistics_episode(config, episode),
                transitions=transitions, successes=successes, successful_actions=successful_actions,
                submissions=submissions, input_violations=invalids, timeouts=timeouts,
                success_rate=successes/config.rollouts, input_violation_rate=invalids/config.rollouts,
                timeout_rate=timeouts/config.rollouts, mean_length=transitions/config.rollouts,
                repeat_pair_visits=repeats, states_visited=len(coverage), new_states=len(coverage)-old_coverage,
                block_states_visited=len(block_coverage), new_block_states=len(block_coverage)-old_block_coverage,
                statistics_rows=len(statistics), statistics_rows_before_episode=before,
                backup_updates=transitions, signed_w_updates=successful_actions-invalids,
                mean_action_success_credit=successful_actions/transitions,
                mean_action_penalty_credit=-invalids/transitions, mean_action_intrinsic_credit=0.0,
                mean_action_credit=(successful_actions-invalids)/transitions,
                mean_rollout_reward=(successes-invalids)/config.rollouts,
                mean_rollout_success_bonus=successes/config.rollouts, mean_rollout_penalty=-invalids/config.rollouts,
                entropy=policy.entropy_sum/policy.decisions, uct_actions=policy.decisions)


def save_checkpoint(path, statistics, config, episode, rng, coverage, block_coverage, schedule, elapsed):
    snapshot = statistics.snapshot()
    atomic_save(dict(algorithm='uct_signed_success_penalty', config=asdict(config),
                               completed_episodes=episode, statistics=snapshot,
                               protected_cells=list(PROTECTED_CELLS), numpy_rng=rng.bit_generator.state,
                               schedule=schedule, current_block=episode, current_question_id=schedule[episode-1],
                               diagnostic_coverage=sorted(coverage), diagnostic_block_coverage=sorted(block_coverage),
                               baseline=None, elapsed_seconds=elapsed), path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('passes', 'rollouts', 'horizon', 'exploration', 'success_value_scale', 'seed', 'log_every', 'checkpoint_every'):
        default = getattr(Config(), name)
        parser.add_argument('--'+name.replace('_', '-'), type=type(default), default=default)
    parser.add_argument('--output', type=Path, help='Output directory; defaults to log/ with seed-specific filenames')
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = Path(__file__).resolve().parent/'log'
    config = Config(**{k: v for k, v in vars(args).items() if k != 'output'})
    if min(config.passes, config.rollouts, config.horizon, config.log_every, config.checkpoint_every) <= 0:
        parser.error('budgets must be positive')
    if not math.isfinite(config.exploration) or config.exploration < 0:
        parser.error('require finite nonnegative exploration')
    if not math.isfinite(config.success_value_scale) or config.success_value_scale <= 0:
        parser.error('require finite positive success-value scale')
    prefix = f'{Path(__file__).stem}_seed{config.seed}'
    if args.output.exists() and any(p.suffix != '.log' or p.stat().st_size for p in args.output.glob(f'{prefix}.*')):
        parser.error('results for this seed already exist; refusing to overwrite previous results')
    return args, config


def main():
    args, config = parse_args()
    prefix = f'{Path(__file__).stem}_seed{config.seed}'
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    rng = np.random.default_rng(config.seed)
    schedule = question_schedule(config)
    statistics = Statistics(config.decay)
    coverage = set()
    print(json.dumps(dict(**asdict(config), algorithm='uct_signed_success_penalty', episodes=config.episodes,
                          schedule_seed=config.seed+1, question_schedule=schedule,
                          statistics_reset='never', statistics_clock='own_question_updates',
                          backup_rule=SIGNED_BACKUP_RULE,
                          value_semantics='successful_action_occurrences_minus_offending_input_violations',
                          selection='success_value_scale*(W/N) + c*sqrt(log(max(1,sum(N)))/N)',
                          value_scale_applies_to='entire_signed_W_including_penalties',
                          unvisited_actions='uniform', tie_breaking='uniform',
                          protected_cells=list(PROTECTED_CELLS), penalty_credit='offending_action_only',
                          violation_write_applied=False, baseline=None, pretraining=None,
                          surprise_weight=0, surprise_computed=False, action_mask=False, neural_training=False,
                          reward_role='signed_W_drives_UCT_exploitation')), flush=True)
    started = time.perf_counter()
    with (args.output/f'{prefix}.metrics.jsonl').open('x', encoding='utf-8', buffering=1) as stream:
        for episode, qid in enumerate(schedule):
            block_coverage = set()
            metrics = collect_episode(statistics, config, episode, rng, coverage, qid, block_coverage)
            elapsed = time.perf_counter()-started
            metrics['elapsed_seconds'] = elapsed
            stream.write(json.dumps(metrics)+'\n')
            if episode == 0 or (episode+1) % config.log_every == 0:
                print(f"episode {episode+1}/{config.episodes} | question {qid//4}+{qid%4} | "
                      f"success {metrics['success_rate']:.3f} | length {metrics['mean_length']:.1f} | "
                      f"states {len(statistics)} | seconds {elapsed:.1f}", flush=True)
            if (episode+1) % config.checkpoint_every == 0 or episode+1 == config.episodes:
                save_checkpoint(args.output/f'{prefix}.checkpoint.pt', statistics, config, episode+1,
                                rng, coverage, block_coverage, schedule, elapsed)
    print('Finished.', flush=True)


if __name__ == '__main__':
    main()
