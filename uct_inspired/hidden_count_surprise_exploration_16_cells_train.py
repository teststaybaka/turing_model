"""Tape + budget with hidden count memory.

Self-contained reproduction of the retained fixed surprise + exploration experiment.
Run without arguments: seed 20261011, 2,250 passes over the original 768 cases,
16 rollouts/update, horizon 64, no pretraining. Use --seed 20261012 or 20261013
for the other saved seeds. The remaining 256 cases are held out in this stage.
Task credit is +1 plus .01 per unused step on every successful action,
-1 on the offending input write, and zero for wrong submit / timeout.
Intrinsic credit is .2*(abs(y-(S-F)/N)-.5) + .01*(1/33-softmax(N)[action]).
y=+1 on successful paths, -1 only on input violations, 0 otherwise.
N/S/F normalize together to total N=3300 after each reverse occurrence backup; no clock decay.
N=100,S=F=50 initially. Rewards stay local to their action occurrences.
Hidden layers 64/64 tanh, Adam .0003, clip 1, no entropy regularizer or critic.
Outputs use this filename as prefix in log/; existing results are never overwritten.
--resume PATH restores an original checkpoint into a separate --output directory.
Only NumPy, PyTorch, and the adjacent data/binary_task_suite_v2_16cells.json are required.
"""

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


INPUT_MODE = 'tape_budget'
INPUT_SIZE = 33
STATISTICS_INPUT = False
MODEL_STATISTICS = []
COUNT_DTYPE = 'float64'
BACKUP_RULE = 'reverse_each_occurrence_then_rescale_all_state_N_S_F'
INTRINSIC_REWARD = 'ratio_surprise_plus_visit'
INTRINSIC_DEFINITION = 'ratio_surprise_weight*(abs(y-(S-F)/N)-ratio_surprise_threshold)+visit_reward_weight*(1/33-softmax(N)[action])'


K, NUM_ACTIONS, SUBMIT = 16, 33, 32
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

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

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

def reinforce_loss(logits, actions, credits, rollouts):
    logps = F.log_softmax(logits, dim=-1)
    selected = logps.gather(1, actions[:, None]).squeeze(1)
    return -(selected * credits.detach()).sum() / rollouts

def schedule_for(config):
    rng = np.random.default_rng(config.seed+1)
    return np.concatenate([rng.permutation(TRAIN_CASES) for _ in range(config.passes)])

@dataclass
class Config:
    passes: int = 2250
    rollouts: int = 16
    horizon: int = 64
    hidden: int = 64
    learning_rate: float = .0003
    success_bonus: float = 1.
    early_success_multiplier: float = .01
    input_change_penalty: float = -1.
    wrong_submission_penalty: float = 0.
    initial_count: float = 100.
    seed: int = 20261011
    log_every: int = 1000
    checkpoint_every: int = 1000
    ratio_surprise_weight: float = .2
    ratio_surprise_threshold: float = .5
    visit_reward_weight: float = .01
    state_count_total: float = 3300.

    @property
    def episodes(self):
        return TRAIN_CONTEXTS*self.passes

    def validate(self):
        if not np.isfinite(list(asdict(self).values())).all():raise ValueError('configuration must be finite')
        if min(self.passes,self.rollouts,self.horizon,self.hidden,self.log_every,self.checkpoint_every)<=0 or self.seed<0:
            raise ValueError('invalid training dimensions or seed')
        if self.learning_rate<=0 or self.initial_count<=0 or min(self.success_bonus,self.early_success_multiplier)<0:
            raise ValueError('invalid learning rate, prior or reward')
        if max(self.input_change_penalty,self.wrong_submission_penalty)>0:raise ValueError('penalties must be nonpositive')
        if min(self.ratio_surprise_weight,self.visit_reward_weight)<0 or not 0<=self.ratio_surprise_threshold<=2:
            raise ValueError('invalid intrinsic scale or threshold')
        if not 0<self.state_count_total<2**53:raise ValueError('invalid normalized count total')


class Statistics:
    """Per-tape N/S/F. The prior is installed once at the first state visit."""
    def __init__(self, config):
        self.config = config
        self.counts = np.zeros((STATES, 2, NUM_ACTIONS), dtype=COUNT_DTYPE)
        self.failures = np.zeros((STATES, NUM_ACTIONS), dtype=COUNT_DTYPE)
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
            self.counts[code, 0] = self.config.initial_count
            self.counts[code, 1] = self.config.initial_count/2
            self.failures[code] = self.config.initial_count/2
            factor=self.config.state_count_total/(NUM_ACTIONS*self.config.initial_count)
            self.counts[code]*=factor
            self.failures[code]*=factor
            self.birth_ages[code] = age
        elif age < previous:
            raise ValueError('statistics age moved backwards')
        
        self.ages[code] = age
        return code, self.counts[code]

    def backup_normalized(self, path, success, invalid, threshold):
        """Reverse occurrence backups; return local reward components in path order."""
        if not path or (success and invalid):
            raise ValueError('invalid normalized backup')
        rewards = np.empty((len(path),2), dtype=np.float64)
        removed = np.zeros(3, dtype=np.float64)
        for step in range(len(path)-1,-1,-1):
            idx, action = path[step]
            values, failures = self.counts[idx], self.failures[idx]
            visit_probability = float(count_softmax(values[0])[action])
            ratio = float((values[1,action]-failures[action])/values[0,action]) if values[0,action] else 0.
            outcome = 1. if success else -1. if invalid and step == len(path)-1 else 0.
            rewards[step] = [abs(outcome-ratio)-threshold, 1./NUM_ACTIONS-visit_probability]
            values[0,action] += 1.
            values[1,action] += float(success)
            if invalid and step == len(path)-1:
                failures[action] += 1.
            before = np.array([values[0].sum(),values[1].sum(),failures.sum()])
            factor = self.config.state_count_total/before[0]
            values *= factor
            failures *= factor
            removed += before-np.array([values[0].sum(),values[1].sum(),failures.sum()])
        return rewards, removed

    def snapshot(self):
        keys = np.flatnonzero(self.ages >= 0)
        result = dict(keys=keys, values=self.counts[keys].copy(), failure_counts=self.failures[keys].copy(),
                      ages=self.ages[keys].copy(), birth_ages=self.birth_ages[keys].copy(),
                      initial_count=self.config.initial_count, prior_initialization='first_state_visit',
                      backup_rule=BACKUP_RULE, value_columns=['N','S'], model_value='S-F',
                      failure_semantics='offending_action_only')
        result['state_count_total']=self.config.state_count_total
        return result


def count_softmax(n):
    """Literal softmax(N), temperature one, evaluated safely in float64."""
    n = np.asarray(n, dtype=np.float64)
    if not np.isfinite(n).all() or (n < 0).any():
        raise ValueError('counts must be finite and nonnegative')
    with np.errstate(under='ignore'):
        weights = np.exp(n-n.max(axis=-1, keepdims=True))
    return weights/weights.sum(axis=-1, keepdims=True)

def encode_features(tapes, remaining_budget):
    size = len(tapes)
    x = np.zeros((size, INPUT_SIZE), dtype=np.float32)
    x[np.arange(size)[:,None], 2*np.arange(K)[None,:]+tapes] = 1
    
    budget = np.asarray(remaining_budget)
    if not np.isfinite(budget).all() or (budget <= 0).any() or (budget > 1).any():
        raise ValueError('pre-action remaining budget must be in (0,1]')
    x[:,-1] = budget
    return x

class Controller(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(INPUT_SIZE,hidden),nn.Tanh(),nn.Linear(hidden,hidden),
                                 nn.Tanh(),nn.Linear(hidden,NUM_ACTIONS))
        nn.init.normal_(self.net[-1].weight,std=.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self,x):
        return self.net(x)

def initialize(config):
    config.validate()
    torch.manual_seed(config.seed)
    model = Controller(config.hidden)
    return (model,torch.optim.Adam(model.parameters(),lr=config.learning_rate),Statistics(config),
            np.random.default_rng(config.seed),np.random.default_rng(config.seed+4))


def collect_episode(model, stats, config, episode, rng, init_rng, coverage, case_id, action_policy=None, statistics_episode=None):
    if stats.counts.dtype != np.float64:
        raise ValueError('normalized reward requires no decay and float64 counts')
    intrinsic_sum = 0.0
    combined_raw_totals = np.zeros(2)
    age = episode // TRAIN_CONTEXTS if statistics_episode is None else statistics_episode
    if not isinstance(age, (int, np.integer)) or age < 0:
        raise ValueError('statistics_episode must be a nonnegative integer')
    xs, actions, credits = ([], [], [])
    successes = submissions = violations = successful_actions = repeats = 0
    success_remaining_steps = successful_actions_remaining_steps = 0
    initial_correct = successful_initial_correct = successful_initial_wrong = 0
    first_action_submissions = first_action_successes = 0
    positive = negative = 0
    normalization_removed = np.zeros(3)
    before = len(coverage)
    rows_before = len(stats)
    block = set()
    for _ in range(config.rollouts):
        tape = Tape(case_id, init_rng)
        started_correct = bool(np.array_equal(tape.cells[PROTECTED_BITS:], tape.expected))
        initial_correct += int(started_correct)
        path, seen = ([], set())
        success = submitted = invalid = False
        with torch.inference_mode():
            for step in range(config.horizon):
                idx, _ = stats.row(tape.code, age)
                x = encode_features(tape.cells[None, :], (config.horizon - step) / config.horizon)[0]
                if action_policy is None:
                    probs = torch.softmax(model(torch.from_numpy(x)), dim=-1).numpy()
                    action = min(int(np.searchsorted(np.cumsum(probs), rng.random())), NUM_ACTIONS - 1)
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
        intrinsic, removed = stats.backup_normalized(path, success, invalid, config.ratio_surprise_threshold)
        normalization_removed += removed
        combined_raw_totals += intrinsic.sum(axis=0)
        intrinsic = config.ratio_surprise_weight * intrinsic[:, 0] + config.visit_reward_weight * intrinsic[:, 1]
        intrinsic_sum += float(intrinsic.sum())
        remaining = config.horizon - len(path)
        success_credit = (config.success_bonus + config.early_success_multiplier * remaining) * int(success)
        local = success_credit + intrinsic
        if invalid:
            local[-1] += config.input_change_penalty
        if submitted and (not success):
            local[-1] += config.wrong_submission_penalty
        credits.extend(local.tolist())
        positive += int((intrinsic > 1e-12).sum())
        negative += int((intrinsic < -1e-12).sum())
        successes += int(success)
        submissions += int(submitted)
        violations += int(invalid)
        successful_actions += len(path) * int(success)
        success_remaining_steps += remaining * int(success)
        successful_actions_remaining_steps += len(path) * remaining * int(success)
        successful_initial_correct += int(success and started_correct)
        successful_initial_wrong += int(success and (not started_correct))
        first_action_submissions += int(submitted and len(path) == 1)
        first_action_successes += int(success and len(path) == 1)
    transitions = len(actions)
    timeouts = config.rollouts - submissions - violations
    metrics = dict(
        episode=episode + 1,
        case_id=case_id,
        operator_id=case_id // INPUT_VALUES,
        input_value=case_id % INPUT_VALUES,
        pass_index=age + 1,
        statistics_episode=age,
        transitions=transitions,
        successes=successes,
        submissions=submissions,
        input_violations=violations,
        timeouts=timeouts,
        successful_actions=successful_actions,
        success_remaining_steps=success_remaining_steps,
        successful_actions_remaining_steps=successful_actions_remaining_steps,
        success_rate=successes / config.rollouts,
        timeout_rate=timeouts / config.rollouts,
        input_violation_rate=violations / config.rollouts,
        mean_length=transitions / config.rollouts,
        repeat_pair_visits=repeats,
        states_visited=len(coverage),
        new_states=len(coverage) - before,
        block_states_visited=len(block),
        new_block_states=len(block),
        statistics_rows=len(stats),
        statistics_rows_before_episode=rows_before,
        backup_updates=transitions,
        success_count_updates=successful_actions,
        failure_count_updates=violations,
        initial_correct=initial_correct,
        successful_initial_correct=successful_initial_correct,
        successful_initial_wrong=successful_initial_wrong,
        first_action_submissions=first_action_submissions,
        first_action_successes=first_action_successes,
        mean_action_success_credit=(config.success_bonus * successful_actions + config.early_success_multiplier * successful_actions_remaining_steps) / transitions,
        mean_action_base_success_credit=config.success_bonus * successful_actions / transitions,
        mean_action_early_success_credit=config.early_success_multiplier * successful_actions_remaining_steps / transitions,
        mean_action_penalty_credit=config.input_change_penalty * violations / transitions,
        mean_action_wrong_submission_credit=config.wrong_submission_penalty * (submissions - successes) / transitions,
        mean_action_intrinsic_credit=intrinsic_sum / transitions,
        mean_action_credit=float(np.mean(credits)),
        mean_rollout_probability_change=0.0,
        mean_rollout_surprise_threshold=0.0,
        mean_rollout_intrinsic_reward=intrinsic_sum / config.rollouts,
        mean_rollout_weighted_intrinsic_reward=intrinsic_sum / config.rollouts,
        mean_rollout_reward=(intrinsic_sum + config.success_bonus * successes + config.early_success_multiplier * success_remaining_steps + config.input_change_penalty * violations + config.wrong_submission_penalty * (submissions - successes)) / config.rollouts,
        positive_intrinsic_edges=positive,
        negative_intrinsic_edges=negative,
    )
    metrics.update(
        intrinsic_reward='ratio_surprise_plus_visit',
        mean_action_surprise_credit=config.ratio_surprise_weight * combined_raw_totals[0] / transitions,
        mean_action_visit_count_credit=config.visit_reward_weight * combined_raw_totals[1] / transitions,
        mean_rollout_raw_ratio_surprise=combined_raw_totals[0] / config.rollouts,
        mean_rollout_raw_visit_reward=combined_raw_totals[1] / config.rollouts,
        intrinsic_diagnostic_definition='sum_of_separately_weighted_surprise_and_visit_credit',
        ratio_surprise_weight=config.ratio_surprise_weight,
        ratio_surprise_threshold=config.ratio_surprise_threshold,
        visit_reward_weight=config.visit_reward_weight,
        visit_count_threshold=1.0 / NUM_ACTIONS,
        surprise_visit_scaling=False,
        counts_recorded='reverse_sequential_rollout_backup',
    )
    metrics.update(zip(('normalized_away_N', 'normalized_away_S', 'normalized_away_F'), normalization_removed.tolist()))
    metrics['state_count_total'] = config.state_count_total
    # Keep historical diagnostic columns, zero because this reward has no probability-change term.
    for name in ('success', 'failure'):
        metrics[f'mean_rollout_{name}_probability_change'] = 0.0
        metrics[f'mean_rollout_{name}_surprise_threshold'] = 0.0
    return (dict(features=torch.from_numpy(np.stack(xs)), actions=torch.tensor(actions, dtype=torch.long), credits=torch.tensor(credits, dtype=torch.float32)), metrics, block)

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

def config_from_checkpoint(cp):
    """Accept historical saved configs only when their algorithm matches this script."""
    saved = cp['config']
    required = dict(input_mode=INPUT_MODE, intrinsic_reward=INTRINSIC_REWARD,
                    pretrain_steps=0, decay_scope='all_pairs', count_dtype=COUNT_DTYPE)
    for key, expected in required.items():
        if saved.get(key, cp.get(key, expected)) != expected:
            raise ValueError('checkpoint algorithm mismatch: '+key)
    if saved.get('decay',1.)!=1. or saved.get('surprise_weight',0.)!=0.:raise ValueError('not the normalized combined-reward setup')
    cfg = Config(**{f.name:saved[f.name] for f in fields(Config) if f.name in saved})
    cfg.validate()
    return cfg

def restore_checkpoint(cp):
    cfg = config_from_checkpoint(cp)
    completed = cp['completed_episodes']
    if 'case_updates' in cp or not 0 < completed <= cfg.episodes:
        raise ValueError('expected original-stage checkpoint')
    np.testing.assert_array_equal(cp['schedule'], schedule_for(cfg))
    visits = np.bincount(cp['schedule'][:completed],minlength=CONTEXTS)
    model, optimizer, stats, rng, init_rng = initialize(cfg)
    model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer'])
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert len(optimizer.state)==6
    for state in optimizer.state.values():
        assert int(state['step'])==completed
        assert all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v))
    snap = cp['statistics'];keys = np.asarray(snap['keys'])
    assert np.issubdtype(keys.dtype,np.integer) and len(np.unique(keys))==len(keys)
    assert ((keys>=0)&(keys<STATES)).all() and (visits[keys//64]>0).all()
    assert snap['initial_count']==cfg.initial_count and snap['backup_rule']==BACKUP_RULE
    assert snap.get('decay_scope','all_pairs')=='all_pairs'
    assert (snap['ages']>=0).all() and (snap['ages']<visits[keys//64]).all()
    assert (snap['birth_ages']>=0).all() and (snap['birth_ages']<=snap['ages']).all()
    assert snap['values'].dtype==stats.counts.dtype and snap['failure_counts'].dtype==stats.failures.dtype
    stats.counts[keys]=snap['values'];stats.failures[keys]=snap['failure_counts']
    stats.ages[keys]=snap['ages'];stats.birth_ages[keys]=snap['birth_ages'];stats.size=len(keys)
    assert snap['state_count_total']==cfg.state_count_total
    np.testing.assert_allclose(stats.counts[keys,0].sum(axis=1),cfg.state_count_total,rtol=1e-12,atol=1e-10)
    assert np.isfinite(stats.counts).all() and np.isfinite(stats.failures).all()
    assert (stats.counts>=0).all() and (stats.failures>=0).all()
    assert (stats.counts[:,1]<=stats.counts[:,0]).all() and (stats.failures<=stats.counts[:,0]).all()
    coverage=set(cp['diagnostic_coverage'])
    assert coverage==set(map(int,keys))
    rng.bit_generator.state=cp['numpy_rng'];init_rng.bit_generator.state=cp['initialization_rng']
    torch.set_rng_state(cp['torch_rng'])
    return cfg,model,optimizer,stats,rng,init_rng,coverage,visits

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name,default in asdict(Config()).items():
        parser.add_argument('--'+name.replace('_','-'),type=type(default),default=default)
    parser.add_argument('--dataset',type=Path,default=DEFAULT_DATASET)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'log')
    parser.add_argument('--resume',type=Path,help='Continue an unfinished original checkpoint into a separate directory')
    args=parser.parse_args(argv)
    torch.set_num_threads(1)
    _,digest=load_dataset(args.dataset)
    completed=0;elapsed_before=0.;source_hash=None
    if args.resume:
        source_hash=sha256(args.resume)
        cp=torch.load(args.resume,map_location='cpu',weights_only=False)
        if cp['dataset_sha256']!=digest:raise ValueError('dataset mismatch')
        cfg,model,optimizer,stats,rng,init_rng,coverage,visits=restore_checkpoint(cp)
        completed=cp['completed_episodes'];elapsed_before=cp['elapsed_seconds']
        if completed==cfg.episodes:raise ValueError('original schedule already completed')
        if args.output.resolve()==args.resume.resolve().parent:raise ValueError('resume requires separate output')
        del cp
    else:
        cfg=Config(**{k:getattr(args,k) for k in asdict(Config())})
        model,optimizer,stats,rng,init_rng=initialize(cfg);coverage=set()
        visits=np.zeros(CONTEXTS,dtype=np.int64)
    prefix=f'{Path(__file__).stem}_seed{cfg.seed}_stage01_original'
    allowed={prefix+'.stdout.log',prefix+'.stderr.log'}
    if args.output.exists() and any(p.name not in allowed or p.stat().st_size for p in args.output.glob(prefix+'.*')):
        raise ValueError('results exist; refusing overwrite')
    args.output.mkdir(parents=True,exist_ok=True)
    schedule=schedule_for(cfg)
    metadata=dict(config=asdict(cfg),input_mode=INPUT_MODE,input_size=INPUT_SIZE,
                  parameters=sum(p.numel() for p in model.parameters()),
                  statistics_input=STATISTICS_INPUT,model_statistics=MODEL_STATISTICS,
                  intrinsic_reward=INTRINSIC_REWARD,intrinsic_definition=INTRINSIC_DEFINITION,
                  count_storage=COUNT_DTYPE,backup_rule=BACKUP_RULE,
                  pretrain_steps=0,statistics_reset='never',statistics_clock='own_case_updates',
                  statistics_decay_scope='all_pairs',training_cases=TRAIN_CASES,heldout_cases=HELDOUT_CASES,
                  protected_cells=list(PROTECTED),tape_cells=K,actions=NUM_ACTIONS,
                  dataset_sha256=digest,schedule=schedule,schedule_seed=cfg.seed+1,
                  initialization_seed=cfg.seed+4,baseline=None,gradient_clip=1,
                  resume_source=str(args.resume.resolve()) if args.resume else None,
                  resume_source_sha256=source_hash)
    print(json.dumps({k:v for k,v in metadata.items() if k!='schedule'}),flush=True)
    started=time.perf_counter()
    with (args.output/(prefix+'.metrics.jsonl')).open('x',encoding='utf-8',buffering=1) as stream:
        for episode in range(completed,cfg.episodes):
            case=int(schedule[episode])
            metrics,block=train_episode(model,optimizer,stats,cfg,episode,rng,init_rng,coverage,case,
                                         statistics_episode=int(visits[case]))
            visits[case]+=1
            elapsed=elapsed_before+time.perf_counter()-started
            metrics['elapsed_seconds']=elapsed
            stream.write(json.dumps(metrics,allow_nan=False)+'\n')
            if episode==completed or (episode+1)%cfg.log_every==0:
                print(f"episode {episode+1}/{cfg.episodes} | success {metrics['success_rate']:.3f} | length {metrics['mean_length']:.2f}",flush=True)
            if (episode+1)%cfg.checkpoint_every==0 or episode+1==cfg.episodes:
                atomic_save(dict(**metadata,completed_episodes=episode+1,model=model.state_dict(),
                    optimizer=optimizer.state_dict(),statistics=stats.snapshot(),numpy_rng=rng.bit_generator.state,
                    initialization_rng=init_rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                    current_case_id=case,diagnostic_coverage=sorted(coverage),diagnostic_block_coverage=sorted(block),
                    elapsed_seconds=elapsed,pretraining=None),args.output/(prefix+'.checkpoint.pt'))
    if args.resume:assert sha256(args.resume)==source_hash
    atomic_save(dict(**metadata,completed_episodes=cfg.episodes,model=model.state_dict(),pretraining=None),
                args.output/(prefix+'.model.pt'))
    print('Finished.',flush=True)

if __name__=='__main__':
    main()

