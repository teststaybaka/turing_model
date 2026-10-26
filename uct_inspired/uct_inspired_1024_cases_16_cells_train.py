"""UCT-inspired tape/statistics/budget policy on 1,024 cases and 16 binary cells.

Run this script without arguments for seed 20261011: 50,000 synthetic UCT
pretraining steps (256 examples/step), then 2,250 RL passes over 768 original
cases. The other 256 cases are held out from this initial stage. Use --seed
20261012 or --seed 20261013 for the other recorded seeds. Pretraining uses
full-distribution cross-entropy against softmax(10*W/N + exploration),
temperature 1, with uniform priority for unvisited actions; no environment
rollouts or statistics updates occur during pretraining.

RL defaults: 16 rollouts/update, horizon 64, hidden layers 64/64, Adam .0003,
success +1 plus .01 per remaining step, offending input-write penalty -1,
surprise multiplier 1, threshold .5, prior N=100/S=50/F=50, own-case decay .99.
No wrong-submit/timeout penalty, entropy bonus, critic, or baseline.

Use --stage new --checkpoint PATH for 512 passes over the 256 new cases,
or --stage mixed --checkpoint PATH for 512 passes over all 1,024 cases.
These continuations restore weights, optimizer, statistics and RNG; they
never rerun pretraining. --stage-index 3 labels an extra new-case stage.
Continuation model/reward settings come from the checkpoint, not CLI defaults.

Files go directly to log/, prefixed with this script's name, seed, and stage:
stage00_pretrain, stage01_original, stage02_new, stage03_new (extra),
stage04_mixed. Existing results are never overwritten. Stages are explicit;
the no-argument command does not automatically start continuations.

Protected cells0..9 encode operator(5 bits) and input(5 bits); cells10..15
are independently randomized per rollout. Only SUBMIT checks the answer.
Statistics and credit mechanics match binary_task_suite_train.
"""
import argparse
import hashlib
import json
import math
from collections import Counter
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


BACKUP_RULE = 'every_occurrence_N_success_S_offending_violation_F'

METHOD = 'full_distribution_softmax_UCT_imitation'

def path_probability_change(statistics, path, success, invalid, threshold):
    """Absolute smoothed probability changes, with a count-scaled threshold.

    Use pre-backup counts and chronological own-rollout occurrence marginals,
    splitting the combined threshold equally by channel.
    """
    seen, seen_failure = Counter(), Counter()
    changes, thresholds = [], []
    for position, (idx, action) in enumerate(path):
        pair = (idx, action)
        values = statistics.values(idx)
        n = float(values[0, action])+seen[pair]
        s = float(values[1, action])+seen[pair]*int(success)
        f = float(statistics.failure_values(idx)[action])+seen_failure[pair]
        yf = bool(invalid and position == len(path)-1)
        probabilities = np.array([(s+1)/(n+2), (f+1)/(n+2)])
        changes.append(np.abs(np.array([success, yf], dtype=float)-probabilities)/(n+3))
        thresholds.append([threshold/(2*(n+3))]*2)
        seen[pair] += 1
        seen_failure[pair] += int(yf)
    return np.asarray(changes), np.asarray(thresholds)

def reinforce_loss(logits, actions, credits, rollouts):
    logps = F.log_softmax(logits, dim=-1)
    selected = logps.gather(1, actions[:, None]).squeeze(1)
    return -(selected * credits.detach()).sum() / rollouts

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

def validate_pretraining_config(config):
    if config.pretrain_steps < 0 or config.pretrain_batch_size <= 0:
        raise ValueError('pretrain steps must be nonnegative and batch size positive')
    positive = (config.pretrain_learning_rate, config.pretrain_temperature,
                config.pretrain_value_scale, config.pretrain_count_min, config.pretrain_count_max)
    if not all(math.isfinite(x) and x > 0 for x in positive):
        raise ValueError('pretraining learning rate, temperature, value scale and count bounds must be finite and positive')
    if config.pretrain_count_max < config.pretrain_count_min:
        raise ValueError('pretrain count maximum must be at least the minimum')
    if not math.isfinite(config.pretrain_exploration) or config.pretrain_exploration < 0:
        raise ValueError('pretrain exploration must be finite and nonnegative')

def uct_distribution(n, w, exploration=1.0, value_scale=10.0, temperature=1.0):
    """Softmax UCT on visited rows; uniform unvisited priority on other rows.

Temperature softens the existing signed UCT score, not the stored statistics.
"""
    n, w = np.asarray(n, dtype=np.float64), np.asarray(w, dtype=np.float64)
    if (n.ndim != 2 or n.shape != w.shape or n.shape[1] < 1 or
            not np.isfinite(n).all() or not np.isfinite(w).all() or
            np.any(n < 0) or np.any(np.abs(w) > n)):
        raise ValueError('expected finite [batch,actions] arrays with N>=0 and |W|<=N')
    if (not math.isfinite(temperature) or temperature <= 0 or
            not math.isfinite(value_scale) or value_scale <= 0 or
            not math.isfinite(exploration) or exploration < 0):
        raise ValueError('require positive temperature/value scale and nonnegative exploration')
    zeros = n == 0
    unvisited = zeros.any(axis=1)
    probabilities = np.zeros_like(n)
    probabilities[unvisited] = zeros[unvisited]/zeros[unvisited].sum(axis=1, keepdims=True)
    active = ~unvisited
    if active.any():
        counts, values = n[active], w[active]
        scores = value_scale*(values/counts)+exploration*np.sqrt(
            np.log(np.maximum(1., counts.sum(axis=1, keepdims=True)))/counts)
        with np.errstate(over='ignore', under='ignore'):
            logits = (scores-scores.max(axis=1, keepdims=True))/temperature
            weights = np.exp(logits)
        probabilities[active] = weights/weights.sum(axis=1, keepdims=True)
    return probabilities

def distribution_loss(logits, probabilities):
    """Batch-mean cross-entropy across all teacher action probabilities."""
    target = torch.as_tensor(probabilities, dtype=logits.dtype, device=logits.device).detach()
    return -(target*F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

def distribution_metrics(logits, probabilities):
    """Teacher entropy, full cross-entropy, and KL for the same input batch."""
    logp = F.log_softmax(logits.detach(), dim=-1).double().numpy()
    positive = probabilities > 0
    teacher_entropy = -float(np.sum(probabilities[positive]*np.log(probabilities[positive]))/len(probabilities))
    cross_entropy = -float(np.sum(probabilities*logp)/len(probabilities))
    return dict(teacher_entropy=teacher_entropy, teacher_cross_entropy=cross_entropy,
                teacher_kl=cross_entropy-teacher_entropy)

def pretrain(model, config, on_step=None, batch_fn=None):
    """Train weights against the full UCT distribution with a separate optimizer.

Independent RNGs leave the RL rollout RNG and question schedule untouched.
Only synthetic inputs are sampled; no target actions are sampled or executed.
RL's optimizer is not passed here, so its state remains fresh.
"""
    validate_pretraining_config(config)
    make_batch = teacher_batch if batch_fn is None else batch_fn
    if config.pretrain_steps == 0:
        return None
    rng = np.random.default_rng(config.seed+2)
    validation_rng = np.random.default_rng(config.seed+3)
    validation_x, validation_probabilities = make_batch(validation_rng, config, 2048)
    with torch.no_grad():
        before = distribution_metrics(model(validation_x), validation_probabilities)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.pretrain_learning_rate)
    for step in range(1, config.pretrain_steps+1):
        x, probabilities = make_batch(rng, config)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = distribution_loss(logits, probabilities)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite pretraining loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        metrics = dict(pretrain_step=step, examples_seen=step*config.pretrain_batch_size,
                       distribution_loss=float(loss.detach()), gradient_norm=float(norm),
                       **distribution_metrics(logits, probabilities))
        if on_step is not None:
            on_step(metrics)
    with torch.no_grad():
        after = distribution_metrics(model(validation_x), validation_probabilities)
    return dict(method=METHOD, completed_steps=config.pretrain_steps,
                examples=config.pretrain_steps*config.pretrain_batch_size,
                synthetic_seed=config.seed+2, validation_seed=config.seed+3, validation_examples=2048,
                validation_before=before, validation_after=after,
                synthetic_distribution='equal_mix_all_zero_partial_unvisited_all_visited; log_uniform_N; uniform_W_over_N_in_minus1_to1; independent_random_valid_tapes',
                rl_optimizer='fresh', rl_statistics='empty', environment_rollouts=0,
                sampled_target_actions=0)


PREFIX = Path(__file__).stem
K, NUM_ACTIONS, SUBMIT = 16, 33, 32
TAPE_INPUT_MODES = ('tape_statistics', 'tape_statistics_budget')
BUDGET_INPUT_MODES = ('statistics_budget', 'tape_statistics_budget')
INPUT_MODES = ('tape_statistics', 'statistics_only', 'statistics_budget', 'tape_statistics_budget')
CONTEXTS, STATES = 1024, 65536
PROTECTED_BITS, ANSWER_BITS, INPUT_VALUES = 10, 6, 32
TRAIN_CONTEXTS = 768
SPLIT_SEED = 20261021
LOOKUP_SEED = 20261022
PROTECTED = tuple(range(PROTECTED_BITS))
POWERS = 2**np.arange(K-1, -1, -1, dtype=np.int64)
DEFAULT_DATASET = Path(__file__).parent/'data'/'binary_task_suite_v2_16cells.json'
OPERATORS = [
    ('ADD', 'A+B, exact; A=abc (0..7), B=de (0..3)'),
    ('SUB_MOD64', '(A-B) modulo64'),
    ('AND', 'A bitwise AND B'),
    ('XOR', 'A bitwise XOR B'),
    ('POPCOUNT5', 'Number of ones in abcde'),
    ('REVERSE5', 'Reverse abcde to edcba, zero extended to6 bits'),
    ('COMPARE', 'Output integer 4*(A<B)+2*(A=B)+(A>B)'),
    ('SORT5', 'Sort abcde ascending, zero extended to6 bits'),
    ('MUL', 'A*B, exact'),
    ('OR', 'A bitwise OR B'),
    ('ROTATE_LEFT5', 'abcde to bcdea'),
    ('CONDITIONAL_REVERSE5', 'If a=0 copy abcde; if a=1 reverse abcde'),
    ('ONE_RUNS5', 'Number of maximal runs of ones in abcde'),
    ('TRANSITIONS5', 'Number of adjacent unequal pairs in abcde'),
    ('LOCAL_PARITIES5', '[a XOR b,b XOR c,c XOR d,d XOR e,e XOR a]'),
    ('GRAY5', 'Binary abcde to Gray code'),
    ('NOT5', 'Bitwise complement of abcde within5 bits'),
    ('SHIFT_LEFT1', 'x*2, exact6-bit output'),
    ('SHIFT_RIGHT1', 'x shifted right by1'),
    ('ROTATE_RIGHT5', 'abcde to eabcd'),
    ('SWAP_ADJACENT5', 'abcde to badce'),
    ('PREFIX_PARITIES5', '[a,a XOR b,a XOR b XOR c,...,a XOR b XOR c XOR d XOR e]'),
    ('SUFFIX_PARITIES5', '[a XOR b XOR c XOR d XOR e,...,d XOR e,e]'),
    ('ONEHOT_MOD6', '1 shifted left by x modulo6'),
    ('SQUARE_MOD64', 'x*x modulo64'),
    ('AFFINE_MOD64', '(13*x+7) modulo64'),
    ('CRC6', 'Start register0; feed abcde MSB first; feedback=register bit5 XOR input; shift left masked63; XOR3 if feedback1'),
    ('HAMMING_DISTANCE_10101', 'Number of differing bits between abcde and10101'),
    ('LONGEST_ONE_RUN5', 'Length of longest contiguous run of ones'),
    ('RANDOM_LOOKUP_A', 'Fixed random6-bit lookup table generated with LOOKUP_SEED'),
    ('RANDOM_LOOKUP_B', 'Second fixed random6-bit lookup table generated with LOOKUP_SEED'),
    ('BIT_PERMUTE5', 'abcde to ceadb'),
]
RANDOM_TABLES = np.random.default_rng(LOOKUP_SEED).integers(0, 64, size=(2, INPUT_VALUES))


def bits(value, width):
    return np.array([(value >> i) & 1 for i in range(width-1, -1, -1)], dtype=np.int64)


def bit_number(values):
    result = 0
    for value in values:
        result = (result << 1) | int(value)
    return result


def oracle(operator, value):
    if not 0 <= operator < 32 or not 0 <= value < INPUT_VALUES:
        raise ValueError('invalid operator/input')
    digits = bits(value, 5).tolist()
    left, right = value >> 2, value & 3
    reverse = bit_number(digits[::-1])
    prefix = [sum(digits[:i+1]) % 2 for i in range(5)]
    suffix = [sum(digits[i:]) % 2 for i in range(5)]
    crc = 0
    for digit in digits:
        feedback = (crc >> 5) ^ digit
        crc = (crc << 1) & 63
        if feedback:
            crc ^= 3
    longest = run = 0
    for digit in digits:
        run = run+1 if digit else 0
        longest = max(longest, run)
    answers = [left+right, (left-right) % 64, left & right, left ^ right,
               sum(digits), reverse, 4*(left < right)+2*(left == right)+(left > right),
               (1 << sum(digits))-1, left*right, left | right,
               ((value << 1) & 31)+(value >> 4), reverse if digits[0] else value,
               sum(digit and (i == 0 or not digits[i-1]) for i,digit in enumerate(digits)),
               sum(digits[i] != digits[i+1] for i in range(4)),
               bit_number([digits[i] ^ digits[(i+1) % 5] for i in range(5)]), value ^ (value >> 1),
               value ^ 31, value << 1, value >> 1, (value >> 1)+((value & 1) << 4),
               bit_number([digits[i] for i in [1,0,3,2,4]]), bit_number(prefix), bit_number(suffix),
               1 << (value % 6), value*value % 64, (13*value+7) % 64, crc,
               sum(a != b for a,b in zip(digits,[1,0,1,0,1])), longest,
               RANDOM_TABLES[0,value], RANDOM_TABLES[1,value],
               bit_number([digits[i] for i in [2,4,0,3,1]])]
    return int(answers[operator])


def case_split():
    rng = np.random.default_rng(SPLIT_SEED)
    heldout = sorted(op*INPUT_VALUES+int(x) for op in range(32)
                     for x in rng.permutation(INPUT_VALUES)[:8])
    excluded = set(heldout)
    train = [case for case in range(CONTEXTS) if case not in excluded]
    return train, heldout


TRAIN_CASES, HELDOUT_CASES = case_split()


def dataset_spec():
    excluded = set(HELDOUT_CASES)
    return dict(version=2, name='binary_task_suite_v2_16cells', tape_cells=K, symbols=[0,1],
                layout=dict(operator=list(range(5)), data=list(range(5,10)), answer=list(range(10,16))),
                protected_cells=list(PROTECTED), bit_order='most_significant_first',
                data_interpretation='abcde; x=16a+8b+4c+2d+e; A=4a+2b+c; B=2d+e',
                initialization='independent_uniform_binary_answer_each_rollout',
                verification='explicit_SUBMIT_only', immediate_submit_baseline=1/64,
                split=dict(seed=SPLIT_SEED, method='8 randomly held-out inputs per operator',
                           training_cases=TRAIN_CASES, heldout_cases=HELDOUT_CASES),
                evaluation_note='Held-out cases never enter RL training. Future transfer tests must freeze network weights and use fresh N100/S50/F50 statistics; no automatic evaluation.',
                random_lookup_seed=LOOKUP_SEED, random_lookup_tables=RANDOM_TABLES.tolist(),
                operators=[dict(operator_id=i, code=f'{i:05b}', name=name, rule=rule)
                           for i,(name,rule) in enumerate(OPERATORS)],
                cases=[dict(case_id=op*INPUT_VALUES+x, operator_id=op, data=x,
                            protected_bits=f'{op:05b}{x:05b}', answer=f'{oracle(op,x):06b}',
                            split='heldout' if op*INPUT_VALUES+x in excluded else 'train')
                       for op in range(32) for x in range(INPUT_VALUES)])


def load_dataset(path):
    raw = path.read_bytes()
    spec = json.loads(raw)
    if spec != dataset_spec():
        raise ValueError('Dataset does not match version2 operator definitions')
    return spec, hashlib.sha256(raw).hexdigest()


class Tape:
    def __init__(self, case_id, init_rng):
        if not 0 <= case_id < CONTEXTS:
            raise ValueError('invalid case')
        self.cells = np.concatenate((bits(case_id, PROTECTED_BITS), init_rng.integers(0, 2, size=ANSWER_BITS)))
        self.code = int(self.cells @ POWERS)
        self.expected = bits(oracle(case_id//INPUT_VALUES, case_id % INPUT_VALUES), ANSWER_BITS)

    def step(self, action):
        if not 0 <= action < NUM_ACTIONS:
            raise ValueError('invalid action')
        if action == SUBMIT:
            return True, bool(np.array_equal(self.cells[PROTECTED_BITS:], self.expected)), False
        position, symbol = divmod(action, 2)
        if position < PROTECTED_BITS and symbol != self.cells[position]:
            return False, False, True
        self.code += int((symbol-self.cells[position])*POWERS[position])
        self.cells[position] = symbol
        return False, False, False


class Statistics:
    """Direct-address table with an optional symmetric prior at first state visit.

    Each action receives N=k,S=F=k/2 once, before its first use. The prior
    subsequently decays along with observed counts on own-case updates.
    """
    def __init__(self, decay, initial_count=0.):
        if not np.isfinite(initial_count) or initial_count < 0:
            raise ValueError('initial count must be finite and nonnegative')
        self.decay = decay
        self.initial_count = initial_count
        self.counts = np.zeros((STATES, 2, NUM_ACTIONS), dtype=np.float32)
        self.failures = np.zeros((STATES, NUM_ACTIONS), dtype=np.float32)
        self.ages = np.full(STATES, -1, dtype=np.int64)
        self.birth_ages = np.full(STATES, -1, dtype=np.int64)
        self.size = 0

    def __len__(self):
        return self.size

    def row(self, code, age):
        if not 0 <= code < STATES or age < 0:
            raise ValueError('invalid state or age')
        previous = int(self.ages[code])
        if previous == -1:
            self.size += 1
            self.counts[code, 0] = self.initial_count
            self.counts[code, 1] = self.initial_count/2
            self.failures[code] = self.initial_count/2
            self.birth_ages[code] = age
        elif age < previous:
            raise ValueError('statistics age moved backwards')
        elif age > previous:
            self.counts[code] *= self.decay**(age-previous)
            self.failures[code] *= self.decay**(age-previous)
        self.ages[code] = age
        return code, self.counts[code]

    def values(self, idx):
        return self.counts[idx]

    def failure_values(self, idx):
        return self.failures[idx]

    def backup(self, path, success, invalid):
        if success and invalid:
            raise ValueError('success and input violation are mutually exclusive')
        for (idx, action), count in Counter(path).items():
            values = self.values(idx)
            values[0, action] += count
            values[1, action] += count*int(success)
        if invalid:
            idx, action = path[-1]
            self.failure_values(idx)[action] += 1

    def snapshot(self):
        keys = np.flatnonzero(self.ages >= 0)
        return dict(keys=keys, values=self.counts[keys].copy(), failure_counts=self.failures[keys].copy(),
                    ages=self.ages[keys].copy(), birth_ages=self.birth_ages[keys].copy(),
                    initial_count=self.initial_count, prior_initialization='first_state_visit',
                    backup_rule=BACKUP_RULE,
                    value_columns=['N', 'S'], model_value='S-F', failure_semantics='offending_action_only')


def input_size(input_mode):
    if input_mode not in INPUT_MODES:
        raise ValueError('invalid input_mode')
    return 2*NUM_ACTIONS+2*K*int(input_mode in TAPE_INPUT_MODES)+int(input_mode in BUDGET_INPUT_MODES)


def encode_features(tapes, n, w, input_mode='tape_statistics', remaining_budget=None):
    size = len(n)
    x = np.zeros((size, input_size(input_mode)), dtype=np.float32)
    offset = 0
    if input_mode in TAPE_INPUT_MODES:
        x[np.arange(size)[:, None], 2*np.arange(K)[None, :]+tapes] = 1
        offset = 2*K
    x[:, offset:offset+NUM_ACTIONS] = np.log1p(n)/8
    x[:, offset+NUM_ACTIONS:offset+2*NUM_ACTIONS] = np.sign(w)*np.log1p(np.abs(w))/8
    if input_mode in BUDGET_INPUT_MODES:
        if remaining_budget is None:
            raise ValueError('budget input modes require normalized remaining budget')
        budget = np.asarray(remaining_budget)
        if not np.isfinite(budget).all() or (budget <= 0).any() or (budget > 1).any():
            raise ValueError('pre-action remaining budget must be in (0,1]')
        x[:, -1] = budget
    return x


class Controller(nn.Module):
    def __init__(self, hidden, input_mode='tape_statistics'):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_size(input_mode), hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                                 nn.Tanh(), nn.Linear(hidden, NUM_ACTIONS))
        nn.init.normal_(self.net[-1].weight, std=.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


@dataclass
class Config:
    input_mode: str = 'tape_statistics_budget'
    passes: int = 2250
    rollouts: int = 16
    horizon: int = 64
    hidden: int = 64
    learning_rate: float = .0003
    success_bonus: float = 1.
    early_success_multiplier: float = .01
    input_change_penalty: float = -1.
    wrong_submission_penalty: float = 0.
    surprise_weight: float = 1.
    probability_change_threshold: float = .5
    decay: float = .99
    initial_count: float = 100.
    seed: int = 20261011
    log_every: int = 1000
    checkpoint_every: int = 1000
    pretrain_steps: int = 50000
    pretrain_batch_size: int = 256
    pretrain_learning_rate: float = .0003
    pretrain_temperature: float = 1.
    pretrain_exploration: float = 1.
    pretrain_value_scale: float = 10.
    pretrain_count_min: float = .001
    pretrain_count_max: float = 102400.

    @property
    def episodes(self):
        return TRAIN_CONTEXTS*self.passes


def teacher_batch(rng, config, size=None):
    size = config.pretrain_batch_size if size is None else size
    tapes = rng.integers(0, 2, size=(size, K))
    n = np.exp(rng.uniform(np.log(config.pretrain_count_min), np.log(config.pretrain_count_max),
                          size=(size, NUM_ACTIONS))).astype(np.float32)
    w = (n*rng.uniform(-1, 1, size=n.shape)).astype(np.float32)
    kinds = rng.integers(0, 3, size=size)
    n[kinds == 0] = 0
    w[kinds == 0] = 0
    for row in np.flatnonzero(kinds == 1):
        unvisited = rng.choice(NUM_ACTIONS, size=int(rng.integers(1, NUM_ACTIONS)), replace=False)
        n[row, unvisited] = 0
        w[row, unvisited] = 0
    target = uct_distribution(n, w, config.pretrain_exploration,
                                       config.pretrain_value_scale, config.pretrain_temperature)
    budget = (rng.integers(1, config.horizon+1, size=size)/config.horizon
              if config.input_mode in BUDGET_INPUT_MODES else None)
    return torch.from_numpy(encode_features(tapes, n, w, config.input_mode, budget)), target


def schedule_for(config):
    rng = np.random.default_rng(config.seed+1)
    return np.concatenate([rng.permutation(TRAIN_CASES) for _ in range(config.passes)])


def initialize(config):
    torch.manual_seed(config.seed)
    model = Controller(config.hidden, config.input_mode)
    return (model, torch.optim.Adam(model.parameters(), lr=config.learning_rate), Statistics(config.decay, config.initial_count),
            np.random.default_rng(config.seed), np.random.default_rng(config.seed+4))


def collect_episode(model, stats, config, episode, rng, init_rng, coverage, case_id, action_policy=None,
                    statistics_episode=None):
    age = episode//TRAIN_CONTEXTS if statistics_episode is None else statistics_episode
    if not isinstance(age, (int, np.integer)) or age < 0:
        raise ValueError('statistics_episode must be a nonnegative integer')
    xs, actions, credits = [], [], []
    successes = submissions = violations = successful_actions = repeats = 0
    success_remaining_steps = successful_actions_remaining_steps = 0
    initial_correct = successful_initial_correct = successful_initial_wrong = 0
    first_action_submissions = first_action_successes = 0
    positive = negative = 0
    changes_total = np.zeros(2)
    thresholds_total = np.zeros(2)
    before = len(coverage)
    rows_before = len(stats)
    block = set()
    for _ in range(config.rollouts):
        tape = Tape(case_id, init_rng)
        started_correct = bool(np.array_equal(tape.cells[PROTECTED_BITS:], tape.expected))
        initial_correct += int(started_correct)
        path, seen = [], set()
        success = submitted = invalid = False
        with torch.inference_mode():
            for step in range(config.horizon):
                idx, values = stats.row(tape.code, age)
                x = encode_features(tape.cells[None, :], values[0:1],
                                    (values[1]-stats.failure_values(idx))[None, :], config.input_mode,
                                    (config.horizon-step)/config.horizon)[0]
                if action_policy is None:
                    probs = torch.softmax(model(torch.from_numpy(x)), dim=-1).numpy()
                    action = min(int(np.searchsorted(np.cumsum(probs), rng.random())), NUM_ACTIONS-1)
                else:
                    action = action_policy(tape)
                coverage.add(tape.code)
                block.add(tape.code)
                pair = (idx, action)
                repeats += int(pair in seen)
                seen.add(pair)
                path.append(pair)
                xs.append(x)
                actions.append(action)
                submitted, success, invalid = tape.step(action)
                if submitted or invalid:
                    break
        changes, thresholds = path_probability_change(stats, path, success, invalid,
                                                             config.probability_change_threshold)
        intrinsic = (changes-thresholds).sum(axis=1)
        remaining = config.horizon-len(path)
        success_credit = (config.success_bonus+config.early_success_multiplier*remaining)*int(success)
        local = success_credit+config.surprise_weight*intrinsic
        if invalid:
            local[-1] += config.input_change_penalty
        if submitted and not success:
            local[-1] += config.wrong_submission_penalty
        stats.backup(path, success, invalid)
        credits.extend(local.tolist())
        changes_total += changes.sum(axis=0)
        thresholds_total += thresholds.sum(axis=0)
        positive += int((intrinsic > 1e-12).sum())
        negative += int((intrinsic < -1e-12).sum())
        successes += int(success)
        submissions += int(submitted)
        violations += int(invalid)
        successful_actions += len(path)*int(success)
        success_remaining_steps += remaining*int(success)
        successful_actions_remaining_steps += len(path)*remaining*int(success)
        successful_initial_correct += int(success and started_correct)
        successful_initial_wrong += int(success and not started_correct)
        first_action_submissions += int(submitted and len(path) == 1)
        first_action_successes += int(success and len(path) == 1)
    transitions = len(actions)
    intrinsic_sum = float((changes_total-thresholds_total).sum())
    timeouts = config.rollouts-submissions-violations
    metrics = dict(episode=episode+1, case_id=case_id, operator_id=case_id//INPUT_VALUES, input_value=case_id % INPUT_VALUES,
                   pass_index=age+1, statistics_episode=age,
                   transitions=transitions, successes=successes, submissions=submissions,
                   input_violations=violations, timeouts=timeouts, successful_actions=successful_actions,
                   success_remaining_steps=success_remaining_steps,
                   successful_actions_remaining_steps=successful_actions_remaining_steps,
                   success_rate=successes/config.rollouts, timeout_rate=timeouts/config.rollouts,
                   input_violation_rate=violations/config.rollouts, mean_length=transitions/config.rollouts,
                   repeat_pair_visits=repeats, states_visited=len(coverage), new_states=len(coverage)-before,
                   block_states_visited=len(block), new_block_states=len(block), statistics_rows=len(stats),
                   statistics_rows_before_episode=rows_before, backup_updates=transitions,
                   success_count_updates=successful_actions, failure_count_updates=violations,
                   initial_correct=initial_correct, successful_initial_correct=successful_initial_correct,
                   successful_initial_wrong=successful_initial_wrong,
                   first_action_submissions=first_action_submissions, first_action_successes=first_action_successes,
                   mean_action_success_credit=(config.success_bonus*successful_actions+
                                              config.early_success_multiplier*successful_actions_remaining_steps)/transitions,
                   mean_action_base_success_credit=config.success_bonus*successful_actions/transitions,
                   mean_action_early_success_credit=config.early_success_multiplier*successful_actions_remaining_steps/transitions,
                   mean_action_penalty_credit=config.input_change_penalty*violations/transitions,
                   mean_action_wrong_submission_credit=config.wrong_submission_penalty*(submissions-successes)/transitions,
                   mean_action_intrinsic_credit=config.surprise_weight*intrinsic_sum/transitions,
                   mean_action_credit=float(np.mean(credits)),
                   mean_rollout_probability_change=float(changes_total.sum()/config.rollouts),
                   mean_rollout_surprise_threshold=float(thresholds_total.sum()/config.rollouts),
                   mean_rollout_intrinsic_reward=intrinsic_sum/config.rollouts,
                   mean_rollout_weighted_intrinsic_reward=config.surprise_weight*intrinsic_sum/config.rollouts,
                   mean_rollout_reward=(config.surprise_weight*intrinsic_sum+config.success_bonus*successes+
                                        config.early_success_multiplier*success_remaining_steps+
                                        config.input_change_penalty*violations+
                                        config.wrong_submission_penalty*(submissions-successes))/config.rollouts,
                   positive_intrinsic_edges=positive, negative_intrinsic_edges=negative)
    for i, name in enumerate(('success', 'failure')):
        metrics[f'mean_rollout_{name}_probability_change'] = float(changes_total[i]/config.rollouts)
        metrics[f'mean_rollout_{name}_surprise_threshold'] = float(thresholds_total[i]/config.rollouts)
    return dict(features=torch.from_numpy(np.stack(xs)), actions=torch.tensor(actions, dtype=torch.long),
                credits=torch.tensor(credits, dtype=torch.float32)), metrics, block


def train_episode(model, optimizer, stats, config, episode, rng, init_rng, coverage, case_id,
                  statistics_episode=None):
    batch, metrics, block = collect_episode(model, stats, config, episode, rng, init_rng, coverage, case_id,
                                           statistics_episode=statistics_episode)
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch['features'])
    loss = reinforce_loss(logits, batch['actions'], batch['credits'], config.rollouts)
    if not torch.isfinite(loss):
        raise RuntimeError('nonfinite RL loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad():
        logps = F.log_softmax(logits, dim=-1)
        entropy = float(-(logps.exp()*logps).sum(-1).mean())
    metrics.update(loss=float(loss.detach()), gradient_norm=float(norm), entropy=entropy,
                   mean_action_advantage=float(batch['credits'].mean()),
                   negative_action_advantage_fraction=float((batch['credits'] < 0).float().mean()))
    return metrics, block


COUNTERS = ('transitions', 'successes', 'submissions', 'input_violations', 'timeouts',
            'successful_actions', 'repeat_pair_visits', 'initial_correct',
            'successful_initial_wrong', 'first_action_submissions')

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def aggregate(rows, rollouts):
    result = {key: sum(row[key] for row in rows) for key in COUNTERS}
    result['updates'] = len(rows)
    result['rollouts'] = len(rows)*rollouts
    result['success_rate'] = result['successes']/result['rollouts']
    result['mean_length'] = result['transitions']/result['rollouts']
    result['successful_mean_length'] = (result['successful_actions']/result['successes']
                                        if result['successes'] else None)
    result['incorrect_submissions'] = result['submissions']-result['successes']
    return result

def restore_checkpoint(checkpoint):
    cfg = Config(**checkpoint['config'])
    completed = checkpoint['completed_episodes']
    if 'case_updates' in checkpoint:
        assert checkpoint['original_training_cases'] == TRAIN_CASES
        assert checkpoint['newly_added_cases'] == HELDOUT_CASES
        visits = np.asarray(checkpoint['case_updates']).copy()
        assert visits.shape == (CONTEXTS,) and np.issubdtype(visits.dtype,np.integer)
        assert (visits >= 0).all()
    else:
        assert checkpoint['training_cases'] == TRAIN_CASES
        assert checkpoint['heldout_cases'] == HELDOUT_CASES
        old_schedule = np.asarray(checkpoint['schedule'])
        assert 0 < completed <= len(old_schedule)
        visits = np.bincount(old_schedule[:completed], minlength=CONTEXTS).astype(np.int64)
        assert (visits[HELDOUT_CASES] == 0).all()
    assert visits.sum() == completed
    model = Controller(cfg.hidden, cfg.input_mode)
    model.load_state_dict(checkpoint['model'])
    model.train().requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer'])
    assert len(optimizer.state) == 6
    assert all(int(v['step'].item()) == completed for v in optimizer.state.values())
    assert all(torch.isfinite(p).all() for p in model.parameters())
    for state in optimizer.state.values():
        assert all(torch.isfinite(value).all() for value in state.values() if torch.is_tensor(value))
    snap = checkpoint['statistics']
    keys = np.asarray(snap['keys'])
    assert np.issubdtype(keys.dtype, np.integer)
    assert len(np.unique(keys)) == len(keys) and ((keys >= 0) & (keys < STATES)).all()
    assert (visits[keys//64] > 0).all()
    assert snap['initial_count'] == cfg.initial_count
    assert snap['backup_rule'] == BACKUP_RULE
    assert (snap['ages'] >= 0).all() and (snap['ages'] < visits[keys//64]).all()
    assert (snap['birth_ages'] >= 0).all() and (snap['birth_ages'] <= snap['ages']).all()
    stats = Statistics(cfg.decay, cfg.initial_count)
    stats.counts[keys] = snap['values']
    stats.failures[keys] = snap['failure_counts']
    stats.ages[keys] = snap['ages']
    stats.birth_ages[keys] = snap['birth_ages']
    stats.size = len(keys)
    assert np.isfinite(stats.counts).all() and np.isfinite(stats.failures).all()
    assert (stats.counts >= 0).all() and (stats.failures >= 0).all()
    assert (stats.counts[:,1] <= stats.counts[:,0]).all()
    assert (stats.failures <= stats.counts[:,0]).all()
    coverage = set(checkpoint['diagnostic_coverage'])
    assert coverage == set(map(int, keys))
    rng, init_rng = np.random.default_rng(), np.random.default_rng()
    rng.bit_generator.state = checkpoint['numpy_rng']
    init_rng.bit_generator.state = checkpoint['initialization_rng']
    torch.set_rng_state(checkpoint['torch_rng'])
    return cfg, model, optimizer, stats, rng, init_rng, coverage, visits

def effective_totals(stats, visits):
    snap = stats.snapshot()
    keys = snap['keys']
    last_ages = visits[keys//64]-1
    assert (last_ages >= snap['ages']).all()
    factor = stats.decay**(last_ages-snap['ages'])
    totals = np.zeros((CONTEXTS,3), dtype=np.float64)
    for col, values in enumerate((snap['values'][:,0],snap['values'][:,1],snap['failure_counts'])):
        np.add.at(totals[:,col], keys//64, values.astype(np.float64).sum(axis=1)*factor)
    return totals

def continuation_schedule(case_group, passes, seed):
    if case_group not in ('all','new'):
        raise ValueError('case group must be all or new')
    cases = np.arange(CONTEXTS) if case_group == 'all' else np.asarray(HELDOUT_CASES)
    rng = np.random.default_rng(seed)
    schedule = np.concatenate([rng.permutation(cases) for _ in range(passes)])
    assert np.array_equal(np.sort(schedule.reshape(-1,len(cases)),axis=1),np.tile(cases,(passes,1)))
    return cases,schedule

def group_summary(rows, group, rollouts):
    selected = [row for row in rows if row['case_group'] == group]
    return aggregate(selected,rollouts) if selected else None

def run_continuation(args):
    """Continue weights, optimizer, counts and RNG into a separately labeled stage."""
    if args.passes <= 0:
        raise ValueError('passes must be positive')
    torch.set_num_threads(1)
    source_hash = sha256(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    prefix = stage_prefix(checkpoint['config']['seed'], args.stage, args.stage_index)
    allowed = {f'{prefix}.stdout.log', f'{prefix}.stderr.log'}
    if args.output.exists():
        existing = args.output.glob(f'{prefix}.*')
        if any(p.name not in allowed or p.stat().st_size for p in existing):
            raise ValueError('results already exist; refusing to overwrite or automatically resume')
    _, digest = load_dataset(args.dataset)
    assert checkpoint['dataset_sha256'] == digest
    cfg, model, optimizer, stats, rng, init_rng, coverage, visits = restore_checkpoint(checkpoint)
    source_completed = checkpoint['completed_episodes']
    source_elapsed = checkpoint['elapsed_seconds']
    expected = effective_totals(stats, visits)
    initial_visits = visits.copy()
    original_weights = {key: value.clone() for key,value in model.state_dict().items()}
    schedule_seed = cfg.seed+(10 if args.case_group == 'all' else 11)
    active_cases,schedule = continuation_schedule(args.case_group,args.passes,schedule_seed)
    context_count = len(active_cases)
    target_visits = initial_visits.copy()
    target_visits[active_cases] += args.passes
    original_case_set = set(TRAIN_CASES)
    metadata = dict(config=asdict(cfg), source_checkpoint=str(args.checkpoint.resolve()),
                    source_checkpoint_sha256=source_hash, source_completed_episodes=source_completed,
                    source_elapsed_seconds=source_elapsed, continuation_passes=args.passes,
                    continuation_updates=len(schedule), total_training_updates_target=source_completed+len(schedule),
                    dataset_sha256=digest, training_cases=active_cases.tolist(), case_group=args.case_group,
                    original_training_cases=TRAIN_CASES, newly_added_cases=HELDOUT_CASES,
                    heldout_cases=[], training_contexts=context_count,
                    schedule_seed=schedule_seed, statistics_clock='own_case_updates',
                    initial_case_updates=initial_visits.tolist(),
                    weights_restored=True, optimizer_restored=True, statistics_restored=True,
                    rollout_rng_restored=True, initialization_rng_restored=True, torch_rng_restored=True,
                    pretraining_rerun=False, evaluation_statistics_used=False,
                    carried_statistics_rows=len(stats),
                    new_case_initial_statistics=dict(N=cfg.initial_count,S=cfg.initial_count/2,F=cfg.initial_count/2),
                    input_size=input_size(cfg.input_mode), parameters=sum(p.numel() for p in model.parameters()),
                    backup_rule=BACKUP_RULE,
                    interpretation='continual adaptation; former held-out cases are training cases; inactive cases keep statistics without aging')
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_name = f'{prefix}.protocol.json'
    summary_name = f'{prefix}.summary.json'
    (args.output/protocol_name).write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata),flush=True)
    # Save the exact branch point before any new rollouts or optimizer updates.
    def save(completed, elapsed, case, block):
        atomic_save(dict(**metadata, completed_episodes=source_completed+completed,
                                 completed_continuation_updates=completed,
                                 model=model.state_dict(), optimizer=optimizer.state_dict(), statistics=stats.snapshot(),
                                 case_updates=visits.copy(), numpy_rng=rng.bit_generator.state,
                                 initialization_rng=init_rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                                 schedule=schedule, current_case_id=case, diagnostic_coverage=sorted(coverage),
                                 diagnostic_block_coverage=sorted(block), elapsed_seconds=source_elapsed+elapsed,
                                 continuation_elapsed_seconds=elapsed), args.output/f'{prefix}.checkpoint.pt')
    save(0,0,None,set())
    del checkpoint
    started = time.perf_counter()
    pass_rows = []
    last_metrics = None
    with (args.output/f'{prefix}.metrics.jsonl').open('x',encoding='utf-8',buffering=1) as stream, \
         (args.output/f'{prefix}.passes.jsonl').open('x',encoding='utf-8',buffering=1) as pass_stream:
        for episode, case in enumerate(schedule):
            case = int(case)
            metrics, block = train_episode(model,optimizer,stats,cfg,episode,rng,init_rng,coverage,case,
                                             statistics_episode=int(visits[case]))
            visits[case] += 1
            expected[case] *= cfg.decay
            expected[case] += metrics['new_states']*NUM_ACTIONS*np.array([cfg.initial_count,cfg.initial_count/2,cfg.initial_count/2])
            expected[case] += [metrics['transitions'],metrics['successful_actions'],metrics['input_violations']]
            elapsed = time.perf_counter()-started
            metrics.update(pass_index=episode//context_count+1, total_training_updates=source_completed+episode+1,
                           case_group='original' if case in original_case_set else 'new', elapsed_seconds=elapsed)
            stream.write(json.dumps(metrics,allow_nan=False)+'\n')
            pass_rows.append(metrics)
            last_metrics = metrics
            if episode == 0 or (episode+1)%cfg.log_every == 0:
                print(f"update {episode+1}/{len(schedule)} | case {case} ({metrics['case_group']}) | "
                      f"success {metrics['success_rate']:.3f} | length {metrics['mean_length']:.1f} | seconds {elapsed:.1f}",flush=True)
            if (episode+1)%context_count == 0:
                row = dict(pass_index=(episode+1)//context_count, elapsed_seconds=elapsed,
                           all_cases=aggregate(pass_rows,cfg.rollouts),
                           original_cases=group_summary(pass_rows,'original',cfg.rollouts),
                           newly_added_cases=group_summary(pass_rows,'new',cfg.rollouts))
                pass_stream.write(json.dumps(row,allow_nan=False)+'\n')
                pass_rows.clear()
            if (episode+1)%cfg.checkpoint_every == 0 or episode+1 == len(schedule):
                actual = effective_totals(stats,visits)
                np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=.1)
                assert set(map(int,stats.snapshot()['keys'])) == coverage
                assert all(int(s['step'].item())==source_completed+episode+1 for s in optimizer.state.values())
                save(episode+1,elapsed,case,block)
    assert np.array_equal(visits,target_visits)
    assert any(not torch.equal(v,original_weights[k]) for k,v in model.state_dict().items())
    assert sha256(args.checkpoint) == source_hash
    atomic_save(dict(**metadata,completed_episodes=source_completed+len(schedule),
                             model=model.state_dict(),case_updates=visits),args.output/f'{prefix}.model.pt')
    (args.output/summary_name).write_text(json.dumps(dict(completed_updates=len(schedule),
        source_checkpoint_unchanged=True,weights_updated=True,counts_reconstructed=True,
        elapsed_seconds=time.perf_counter()-started,last_metrics=last_metrics),indent=2),encoding='utf-8')
    print('Finished. Continuation verified.',flush=True)


def stage_prefix(seed, stage, stage_index=None):
    default_index = {'pretrain': 0, 'original': 1, 'new': 2, 'mixed': 4}
    index = default_index[stage] if stage_index is None else stage_index
    if index < 0 or (stage != 'pretrain' and index == 0):
        raise ValueError('RL stage index must be positive')
    return f'{PREFIX}_seed{seed}_stage{index:02d}_{stage}'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in asdict(Config()).items():
        parser.add_argument('--'+name.replace('_', '-'), type=type(default),
                            default=None if name == 'passes' else default)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'log')
    parser.add_argument('--stage', choices=('original', 'new', 'mixed'), default='original')
    parser.add_argument('--stage-index', type=int, help='Stage number in output filenames')
    parser.add_argument('--checkpoint', type=Path, help='Source checkpoint for new/mixed stages')
    args = parser.parse_args(argv)
    if args.passes is None:
        args.passes = Config().passes if args.stage == 'original' else 512
    if (args.stage == 'original') == (args.checkpoint is not None):
        parser.error('original starts fresh; new/mixed require --checkpoint')
    if args.stage_index is not None and args.stage_index < 1:
        parser.error('stage-index must be positive')
    cfg = Config(**{k: getattr(args, k) for k in asdict(Config())})
    if cfg.input_mode not in INPUT_MODES:
        parser.error('input-mode must be one of: '+', '.join(INPUT_MODES))
    validate_pretraining_config(cfg)
    if min(cfg.passes, cfg.rollouts, cfg.horizon, cfg.hidden, cfg.log_every, cfg.checkpoint_every) <= 0:
        parser.error('budgets and dimensions must be positive')
    scales = [cfg.learning_rate, cfg.success_bonus, cfg.input_change_penalty,
              cfg.surprise_weight, cfg.probability_change_threshold, cfg.decay, cfg.initial_count,
              cfg.wrong_submission_penalty, cfg.early_success_multiplier]
    if (not np.isfinite(scales).all() or cfg.learning_rate <= 0 or cfg.success_bonus < 0 or
            cfg.input_change_penalty > 0 or cfg.wrong_submission_penalty > 0 or cfg.early_success_multiplier < 0 or cfg.surprise_weight < 0 or cfg.decay != .99 or
            not 0 <= cfg.probability_change_threshold <= 2 or cfg.initial_count < 0 or cfg.seed < 0):
        parser.error('invalid reward, decay, threshold, or seed')
    if args.stage == 'original':
        prefixes = [stage_prefix(cfg.seed, args.stage, args.stage_index)]
        if cfg.pretrain_steps:
            prefixes.append(stage_prefix(cfg.seed, 'pretrain'))
        for prefix in prefixes:
            allowed = {f'{prefix}.stdout.log', f'{prefix}.stderr.log'}
            if args.output.exists() and any(p.name not in allowed or p.stat().st_size
                                           for p in args.output.glob(f'{prefix}.*')):
                parser.error('results for this seed/stage already exist; refusing to overwrite')
    return args, cfg


def main(argv=None):
    args, cfg = parse_args(argv)
    if args.stage != 'original':
        args.case_group = 'new' if args.stage == 'new' else 'all'
        return run_continuation(args)
    prefix = stage_prefix(cfg.seed, args.stage, args.stage_index)
    pretrain_prefix = stage_prefix(cfg.seed, 'pretrain')
    dataset, digest = load_dataset(args.dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    model, optimizer, stats, rng, init_rng = initialize(cfg)
    schedule = schedule_for(cfg)
    coverage = set()
    metadata = dict(config=asdict(cfg), episodes=cfg.episodes, input_size=input_size(cfg.input_mode),
                    raw_tape_input=cfg.input_mode in TAPE_INPUT_MODES,
                    remaining_budget_input=cfg.input_mode in BUDGET_INPUT_MODES,
                    budget_input_definition='(horizon-actions_already_taken)/horizon_before_action',
                    success_reward_definition='success_bonus+early_success_multiplier*(horizon-rollout_length_including_SUBMIT)',
                    early_success_statistics='binary_S_updates_unchanged',
                    parameters=sum(p.numel() for p in model.parameters()), actions=NUM_ACTIONS,
                    tape_cells=K, tape_symbols=2, protected_cells=list(PROTECTED),
                    dataset_sha256=digest, dataset_path=str(args.dataset.resolve()),
                    contexts=CONTEXTS, training_contexts=TRAIN_CONTEXTS, heldout_contexts=len(HELDOUT_CASES),
                    training_cases=TRAIN_CASES, heldout_cases=HELDOUT_CASES, split_seed=SPLIT_SEED,
                    maximum_statistics_rows=STATES, maximum_training_statistics_rows=TRAIN_CONTEXTS*2**ANSWER_BITS, statistics_reset='never',
                    statistics_clock='own_case_updates', model_statistics=['N', 'S-F'],
                    backup_rule=BACKUP_RULE,
                    initial_statistics=dict(N=cfg.initial_count, S=cfg.initial_count/2,
                                            F=cfg.initial_count/2, applied='first_state_visit_all_actions'),
                    initialization='uniform_random_six_answer_bits_each_rollout',
                    schedule_seed=cfg.seed+1, initialization_seed=cfg.seed+4,
                    schedule_first_pass=schedule[:TRAIN_CONTEXTS].tolist(), schedule_last_case=int(schedule[-1]),
                    surprise='(abs(yS-pS)+abs(yF-pF)-b)/(N+3)', baseline=None,
                    answer_verification='SUBMIT_only', violation_write_applied=False,
                    penalty_credit='offending_action_only', success_credit='every_successful_rollout_action',
                    wrong_submission_credit='incorrect_SUBMIT_action_only',
                    wrong_submission_statistics='N_increment_only_no_S_or_F_increment',
                    entropy_regularization=0, gradient_clip=1, loss_normalization='rollouts_not_actions')
    print(json.dumps(metadata), flush=True)
    pretraining = None
    if cfg.pretrain_steps:
        started = time.perf_counter()
        with (args.output/f'{pretrain_prefix}.metrics.jsonl').open('x', encoding='utf-8', buffering=1) as stream:
            def log_pretrain(metrics):
                metrics['elapsed_seconds'] = time.perf_counter()-started
                stream.write(json.dumps(metrics, allow_nan=False)+'\n')
                if metrics['pretrain_step'] == 1 or metrics['pretrain_step'] % cfg.log_every == 0:
                    print(f"pretrain {metrics['pretrain_step']}/{cfg.pretrain_steps} | KL {metrics['teacher_kl']:.5f}", flush=True)
            pretraining = pretrain(model, cfg, log_pretrain, batch_fn=teacher_batch)
        pretraining['elapsed_seconds'] = time.perf_counter()-started
        pretraining['synthetic_distribution'] += '; all16_tape_cells_binary;33_actions; no_task_oracle_used'
        pretraining['rl_statistics'] = metadata['initial_statistics']
        pretraining['remaining_budget_input'] = ('uniform_integer_1_to_horizon_normalized; UCT_target_independent_of_budget'
                                                if cfg.input_mode in BUDGET_INPUT_MODES else 'absent')
        assert len(stats) == 0 and not optimizer.state
        atomic_save(dict(**metadata, completed_episodes=0, model=model.state_dict(), pretraining=pretraining),
                          args.output/f'{pretrain_prefix}.model.pt')
        print(json.dumps(dict(pretraining_complete=pretraining)), flush=True)
    started = time.perf_counter()
    with (args.output/f'{prefix}.metrics.jsonl').open('x', encoding='utf-8', buffering=1) as stream:
        for episode, case in enumerate(schedule):
            case = int(case)
            metrics, block = train_episode(model, optimizer, stats, cfg, episode, rng, init_rng, coverage, case)
            elapsed = time.perf_counter()-started
            metrics['elapsed_seconds'] = elapsed
            stream.write(json.dumps(metrics, allow_nan=False)+'\n')
            if episode == 0 or (episode+1) % cfg.log_every == 0:
                print(f"episode {episode+1}/{cfg.episodes} | case {case} | success {metrics['success_rate']:.3f} | length {metrics['mean_length']:.1f} | seconds {elapsed:.1f}", flush=True)
            if (episode+1) % cfg.checkpoint_every == 0 or episode+1 == cfg.episodes:
                atomic_save(dict(**metadata, completed_episodes=episode+1, model=model.state_dict(),
                                       optimizer=optimizer.state_dict(), statistics=stats.snapshot(),
                                       numpy_rng=rng.bit_generator.state, initialization_rng=init_rng.bit_generator.state,
                                       torch_rng=torch.get_rng_state(), schedule=schedule, current_case_id=case,
                                       diagnostic_coverage=sorted(coverage), diagnostic_block_coverage=sorted(block),
                                       elapsed_seconds=elapsed, pretraining=pretraining),
                                  args.output/f'{prefix}.checkpoint.pt')
    atomic_save(dict(**metadata, completed_episodes=cfg.episodes, model=model.state_dict(),
                           schedule=schedule, pretraining=pretraining), args.output/f'{prefix}.model.pt')
    print('Finished.', flush=True)


if __name__ == '__main__':
    main()
