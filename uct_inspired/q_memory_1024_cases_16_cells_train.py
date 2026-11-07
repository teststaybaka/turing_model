"""Q-memory policy experiment on the existing 1,024-case binary tape suite.

Fresh weights, no pretraining; train on the original 768 cases for 2,250 passes.
Policy input: tape one-hot (32), raw Q values (33), softmax visit counts (33),
and remaining budget (1). The tape environment and optimizer utilities are implemented inline.
Run without arguments for seed 20261011; --seed selects either other seed.
Outputs use this script's prefix under log/ and are never overwritten.

After EACH rollout, process its Q transitions ONCE from end to start:
  target = task_reward + discount * cached_max_Q(next_tape)
  Q += q_learning_rate * (target - Q)
Terminal transitions never bootstrap. Q stores only task return: correct SUBMIT
gets success + early-success bonus, offending write gets the violation penalty,
other transitions get zero. The horizon is a true terminal boundary.
Q and counts are keyed by tape only. Remaining budget stays in the policy input
and terminal reward calculation; Q intentionally shares values across budgets.
Cached maxima are updated after both increases and decreases, so earlier visited
transitions can use the new successor maximum immediately. No predecessor sweep,
unvisited-pair updates, surprise in Q targets, or decay/normalization of Q.

The policy receives .2 * actual positive Q increase, otherwise -.1 boring credit,
plus .01 * (1/33 - softmax(N)[action]) computed BEFORE that occurrence's visit
increment, retaining chronological order for visit updates. Counts are keyed by
tape only, start at 100/action, and are normalized
to total 3300 after every occurrence. No separate time decay. No S/F counts.
Policy task credit remains +1 plus early bonus on every successful-rollout action
and -1 only on an offending write. Credits are local, never broadcast/summed over
the path. Policy weights update once per 16 rollouts; memory updates after each.

--resume PATH continues an original-stage checkpoint into a separate directory,
restoring all RNG, memory, weights and Adam state without modifying the source.
New/mixed-case stages and automatic new runs are not part of this script.
"""
import argparse
import hashlib
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F




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

INPUT_SIZE = 99

class Controller(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(INPUT_SIZE,hidden),nn.Tanh(),nn.Linear(hidden,hidden),
                                 nn.Tanh(),nn.Linear(hidden,NUM_ACTIONS))
        nn.init.normal_(self.net[-1].weight,std=.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self,x):
        return self.net(x)



def count_softmax(n):
    """Literal softmax(N), temperature one, evaluated safely in float64."""
    n = np.asarray(n, dtype=np.float64)
    if not np.isfinite(n).all() or (n < 0).any():
        raise ValueError('counts must be finite and nonnegative')
    with np.errstate(under='ignore'):
        weights = np.exp(n-n.max(axis=-1, keepdims=True))
    return weights/weights.sum(axis=-1, keepdims=True)


@dataclass
class Config:
    seed: int = 20261011
    passes: int = 2250
    rollouts: int = 16
    horizon: int = 64
    hidden: int = 64
    learning_rate: float = .0003
    q_learning_rate: float = .8
    discount: float = .99
    q_surprise_weight: float = .2
    boring_penalty: float = -.1
    visit_reward_weight: float = .01
    success_bonus: float = 1.
    early_success_multiplier: float = .01
    input_change_penalty: float = -1.
    initial_count: float = 100.
    checkpoint_every: int = 1000
    log_every: int = 1000

    @property
    def episodes(self):
        return self.passes * TRAIN_CONTEXTS

    def validate(self):
        if min(self.passes, self.rollouts, self.horizon, self.hidden,
               self.checkpoint_every, self.log_every) <= 0 or self.seed < 0:
            raise ValueError('invalid budget, dimensions, or seed')
        if not np.isfinite(list(asdict(self).values())).all():
            raise ValueError('configuration must be finite')
        if not 0 < self.q_learning_rate <= 1 or not 0 <= self.discount <= 1:
            raise ValueError('Q learning rate must be in (0,1], discount in [0,1]')
        if (self.learning_rate <= 0 or self.initial_count <= 0 or
            min(self.q_surprise_weight, self.visit_reward_weight, self.success_bonus,
                self.early_success_multiplier) < 0 or
            max(self.boring_penalty, self.input_change_penalty) > 0):
            raise ValueError('invalid reward scale, count prior, or policy learning rate')


class Memory:
    def __init__(self, config):
        self.config = config
        # Implicit zero rows are allocated only after a nonzero Q update.
        self.q = {}
        self.max_q = {}
        self.zero_q = np.zeros(NUM_ACTIONS, dtype=np.float64)
        self.zero_q.flags.writeable = False
        self.counts = np.zeros((STATES, NUM_ACTIONS), dtype=np.float64)
        self.seen = np.zeros(STATES, dtype=bool)

    def key(self, code, remaining):
        if not 0 <= code < STATES or not 1 <= remaining <= self.config.horizon:
            raise ValueError('invalid tape state or remaining budget')
        return int(code)

    def row(self, code, remaining):
        key = self.key(code, remaining)
        if not self.seen[code]:
            self.counts[code] = self.config.initial_count
            self.seen[code] = True
        return self.q.get(key, self.zero_q), self.counts[code]

    def maximum(self, code, remaining):
        return self.max_q.get(self.key(code, remaining), 0.)

    def update_q(self, code, remaining, action, reward, next_code, terminal):
        """Update only the visited Q pair and its state's cached maximum."""
        cfg = self.config
        if not 0 <= action < NUM_ACTIONS or not np.isfinite(reward):
            raise ValueError('invalid transition action or reward')
        if remaining == 1 and not terminal:
            raise ValueError('exhausted horizon must terminate without bootstrapping')
        q, _ = self.row(code, remaining)
        old = float(q[action])
        target = float(reward)
        if not terminal:
            target += cfg.discount * self.maximum(next_code, remaining-1)
        new = old + cfg.q_learning_rate * (target-old)
        if not np.isfinite(new):
            raise FloatingPointError('nonfinite Q update')
        key = self.key(code, remaining)
        if key in self.q or new != 0.:
            if key not in self.q:
                q = self.zero_q.copy()
                self.q[key] = q
            q[action] = new
            previous_max = self.max_q.get(key, 0.)
            if new >= previous_max:
                self.max_q[key] = new
            elif old == previous_max:
                # Includes tied former maxima and a formerly unique best action.
                self.max_q[key] = float(q.max())
            else:
                self.max_q.setdefault(key, previous_max)
        delta = new-old
        surprise = cfg.q_surprise_weight*delta if delta > 0 else cfg.boring_penalty
        return dict(delta=delta, target=target, surprise=float(surprise))

    def update_visit(self, code, remaining, action):
        cfg = self.config
        _, n = self.row(code, remaining)
        visit = cfg.visit_reward_weight * (1/NUM_ACTIONS-count_softmax(n)[action])
        n[action] += 1.
        n *= (cfg.initial_count*NUM_ACTIONS)/n.sum()
        return float(visit)

    def update_rollout(self, path):
        """One Q update per occurrence; return credits aligned to original actions."""
        results = [None] * len(path)
        order = reversed(range(len(path)))
        for i in order:
            results[i] = self.update_q(*path[i])
        # Counts are independent of Q: preserve their previous chronological semantics.
        for transition, result in zip(path, results):
            result['visit'] = self.update_visit(*transition[:3])
        return results

    def snapshot(self):
        keys = np.array(sorted(self.q), dtype=np.int64)
        return dict(q_keys=keys,
                    q_values=np.stack([self.q[int(k)] for k in keys]) if len(keys)
                    else np.empty((0, NUM_ACTIONS), dtype=np.float64),
                    q_max=np.array([self.max_q[int(k)] for k in keys]),
                    count_keys=np.flatnonzero(self.seen), counts=self.counts[self.seen].copy())

    @classmethod
    def restore(cls, config, snapshot):
        result = cls(config)
        keys, values, maxima = (snapshot[k] for k in ('q_keys', 'q_values', 'q_max'))
        assert np.issubdtype(keys.dtype, np.integer) and len(np.unique(keys)) == len(keys)
        assert values.shape == (len(keys), NUM_ACTIONS) and maxima.shape == (len(keys),)
        assert np.isfinite(values).all() and np.isfinite(maxima).all()
        assert ((keys >= 0) & (keys < STATES)).all()
        np.testing.assert_array_equal(values.max(axis=1), maxima)
        lower = min(0., config.input_change_penalty)
        upper = config.success_bonus+config.early_success_multiplier*(config.horizon-1)
        assert (values >= lower-1e-10).all() and (values <= upper+1e-10).all()
        for key, value, maximum in zip(keys, values, maxima):
            result.q[int(key)] = value.copy()
            result.max_q[int(key)] = float(maximum)
        ckeys, counts = snapshot['count_keys'], snapshot['counts']
        assert np.issubdtype(ckeys.dtype, np.integer) and len(np.unique(ckeys)) == len(ckeys)
        assert ((ckeys >= 0) & (ckeys < STATES)).all()
        assert counts.shape == (len(ckeys), NUM_ACTIONS)
        assert np.isfinite(counts).all() and (counts >= 0).all()
        np.testing.assert_allclose(counts.sum(axis=1), config.initial_count*NUM_ACTIONS,
                                   rtol=1e-12, atol=1e-10)
        assert set(map(int, keys)).issubset(set(map(int, ckeys)))
        result.seen[ckeys] = True
        result.counts[ckeys] = counts
        return result


def features(tape, remaining, memory):
    q, n = memory.row(tape.code, remaining)
    x = np.zeros(99, dtype=np.float32)
    x[2*np.arange(K)+tape.cells] = 1
    x[32:65] = q
    x[65:98] = count_softmax(n)
    x[98] = remaining/memory.config.horizon
    return x


def initialize(cfg):
    cfg.validate()
    torch.manual_seed(cfg.seed)
    model = Controller(cfg.hidden)
    return (model, torch.optim.Adam(model.parameters(), lr=cfg.learning_rate), Memory(cfg),
            np.random.default_rng(cfg.seed), np.random.default_rng(cfg.seed+4))


def collect_episode(model, memory, cfg, episode, rng, init_rng, case_id, action_policy=None):
    xs, actions, credits = [], [], []
    counts = dict(successes=0, submissions=0, input_violations=0, timeouts=0,
                  successful_actions=0, repeat_pair_visits=0, success_remaining_steps=0,
                  successful_actions_remaining_steps=0, first_action_submissions=0,
                  first_action_successes=0, initial_correct=0)
    sums = dict(surprise=0., visit=0., task=0., delta=0., positive_delta=0.)
    increased = unchanged = decreased = 0
    before = int(memory.seen.sum())
    for _ in range(cfg.rollouts):
        tape = Tape(case_id, init_rng)
        counts['initial_correct'] += int(np.array_equal(tape.cells[PROTECTED_BITS:], tape.expected))
        path, seen = [], set()
        submitted = success = invalid = False
        with torch.inference_mode():
            for step in range(cfg.horizon):
                remaining = cfg.horizon-step
                code = tape.code
                x = features(tape, remaining, memory)
                if action_policy is None:
                    probabilities = torch.softmax(model(torch.from_numpy(x)), dim=-1).numpy()
                    action = min(int(np.searchsorted(np.cumsum(probabilities), rng.random())), NUM_ACTIONS-1)
                else:
                    action = action_policy(tape)
                pair = (code, action)
                counts['repeat_pair_visits'] += int(pair in seen)
                seen.add(pair)
                submitted, success, invalid = tape.step(action)
                terminal = submitted or invalid or remaining == 1
                reward = (cfg.success_bonus+cfg.early_success_multiplier*(remaining-1)
                          if success else cfg.input_change_penalty if invalid else 0.)
                path.append((code, remaining, action, reward, tape.code, terminal))
                xs.append(x); actions.append(action)
                if terminal:
                    break
        # Keep policy credits aligned with the actions, regardless of Q update order.
        success_credit = (cfg.success_bonus+cfg.early_success_multiplier*(cfg.horizon-len(path)))*int(success)
        for i, result in enumerate(memory.update_rollout(path)):
            task_credit = success_credit+(cfg.input_change_penalty if invalid and i == len(path)-1 else 0.)
            credits.append(task_credit+result['surprise']+result['visit'])
            sums['task'] += task_credit
            sums['surprise'] += result['surprise']; sums['visit'] += result['visit']
            sums['delta'] += result['delta']; sums['positive_delta'] += max(0., result['delta'])
            increased += int(result['delta'] > 0)
            unchanged += int(result['delta'] == 0)
            decreased += int(result['delta'] < 0)
        counts['successes'] += int(success); counts['submissions'] += int(submitted)
        counts['input_violations'] += int(invalid)
        counts['timeouts'] += int(not submitted and not invalid)
        counts['successful_actions'] += len(path)*int(success)
        counts['success_remaining_steps'] += (cfg.horizon-len(path))*int(success)
        counts['successful_actions_remaining_steps'] += len(path)*(cfg.horizon-len(path))*int(success)
        counts['first_action_submissions'] += int(submitted and len(path) == 1)
        counts['first_action_successes'] += int(success and len(path) == 1)
    t = len(actions)
    metrics = dict(episode=episode+1, case_id=int(case_id), pass_index=episode//TRAIN_CONTEXTS+1,
                   transitions=t, **counts, success_rate=counts['successes']/cfg.rollouts,
                   mean_length=t/cfg.rollouts, states_visited=int(memory.seen.sum()),
                   new_states=int(memory.seen.sum())-before, q_rows=len(memory.q), q_updates=t,
                   q_increases=increased, q_unchanged=unchanged, q_decreases=decreased,
                   mean_q_change=sums['delta']/t, mean_positive_q_change=sums['positive_delta']/t,
                   mean_action_surprise_credit=sums['surprise']/t,
                   mean_action_visit_count_credit=sums['visit']/t,
                   mean_action_intrinsic_credit=(sums['surprise']+sums['visit'])/t,
                   mean_action_task_credit=sums['task']/t, mean_action_credit=float(np.mean(credits)),
                   mean_rollout_intrinsic_reward=(sums['surprise']+sums['visit'])/cfg.rollouts,
                   mean_rollout_reward=(sums['surprise']+sums['visit']+
                       cfg.success_bonus*counts['successes']+
                       cfg.early_success_multiplier*counts['success_remaining_steps']+
                       cfg.input_change_penalty*counts['input_violations'])/cfg.rollouts)
    return dict(features=torch.from_numpy(np.stack(xs)), actions=torch.tensor(actions, dtype=torch.long),
                credits=torch.tensor(credits, dtype=torch.float32)), metrics


def train_episode(model, optimizer, memory, cfg, episode, rng, init_rng, case_id):
    batch, metrics = collect_episode(model, memory, cfg, episode, rng, init_rng, case_id)
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch['features'])
    loss = reinforce_loss(logits, batch['actions'], batch['credits'], cfg.rollouts)
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite policy loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad():
        logp = F.log_softmax(logits, dim=-1)
        entropy = float(-(logp.exp()*logp).sum(-1).mean())
    metrics.update(loss=float(loss.detach()), gradient_norm=float(norm), entropy=entropy,
                   mean_action_advantage=float(batch['credits'].mean()),
                   negative_action_advantage_fraction=float((batch['credits'] < 0).float().mean()))
    return metrics


def checkpoint(metadata, model, optimizer, memory, rng, init_rng, completed, elapsed):
    return dict(**metadata, completed_episodes=completed, elapsed_seconds=elapsed,
                model=model.state_dict(), optimizer=optimizer.state_dict(), memory=memory.snapshot(),
                numpy_rng=rng.bit_generator.state, initialization_rng=init_rng.bit_generator.state,
                torch_rng=torch.get_rng_state())


def restore(cp):
    saved=cp['config']
    if saved.get('q_key','tape_only')!='tape_only' or cp.get('q_key')!='tape_only':
        raise ValueError('this retained experiment uses tape-only Q')
    if saved.get('q_update_order','reverse')!='reverse' or cp['q_update_order']!='reverse_once_after_each_rollout':
        raise ValueError('this retained experiment uses reverse backups')
    cfg=Config(**{f.name:saved[f.name] for f in fields(Config) if f.name in saved})
    model, optimizer, _, rng, init_rng = initialize(cfg)
    expected_format = 'q_memory_v2_tape'
    assert cp['format']==expected_format and cp['input_size']==99
    assert cp['q_key']=='tape_only' and cp['count_key']=='tape_only'
    assert 0 < cp['completed_episodes'] <= cfg.episodes
    np.testing.assert_array_equal(cp['schedule'], schedule_for(cfg))
    model.load_state_dict(cp['model']); optimizer.load_state_dict(cp['optimizer'])
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert len(optimizer.state)==6
    for state in optimizer.state.values():
        assert int(state['step'])==cp['completed_episodes']
        assert all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v))
    memory = Memory.restore(cfg, cp['memory'])
    assert set(np.flatnonzero(memory.seen)//64).issubset(set(TRAIN_CASES))
    rng.bit_generator.state=cp['numpy_rng'];init_rng.bit_generator.state=cp['initialization_rng']
    torch.set_rng_state(cp['torch_rng'])
    return cfg, model, optimizer, memory, rng, init_rng


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name, default in asdict(Config()).items():
        parser.add_argument('--'+name.replace('_','-'),type=type(default),default=default)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'log')
    parser.add_argument('--dataset',type=Path,default=DEFAULT_DATASET)
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args(argv)
    torch.set_num_threads(1)
    _, digest=load_dataset(args.dataset)
    completed=0;elapsed_before=0.;source_hash=None
    if args.resume:
        source_hash=sha256(args.resume)
        cp=torch.load(args.resume,map_location='cpu',weights_only=False)
        assert cp['dataset_sha256']==digest
        cfg,model,optimizer,memory,rng,init_rng=restore(cp)
        completed=cp['completed_episodes'];elapsed_before=cp['elapsed_seconds']
        if completed==cfg.episodes:
            raise ValueError('checkpoint already completed its original-stage budget')
    else:
        cfg=Config(**{k:getattr(args,k) for k in asdict(Config())})
        model,optimizer,memory,rng,init_rng=initialize(cfg)
    prefix=f'{Path(__file__).stem}_seed{cfg.seed}_stage01_original'
    allowed={prefix+'.stdout.log',prefix+'.stderr.log'}
    if args.output.exists() and any(p.name not in allowed or p.stat().st_size
                                   for p in args.output.glob(prefix+'.*')):
        raise ValueError('results exist; refusing overwrite or duplicate launch')
    args.output.mkdir(parents=True,exist_ok=True)
    schedule=schedule_for(cfg)
    metadata=dict(format='q_memory_v2_tape',config=asdict(cfg),input_size=99,
                  parameters=sum(p.numel() for p in model.parameters()),
                  policy_input=['tape_onehot','raw_Q','softmax_N','remaining_budget'],
                  q_key='tape_only',q_initial=0.,count_key='tape_only',
                  q_key_encoding='tape_code',
                  remaining_budget_semantics='policy_input_and_terminal_early_reward_only; shared_Q_mixes_budgets',
                  q_storage='implicit_zero_rows_materialized_only_on_nonzero_update',
                  q_reward='terminal_task_reward_only; no intrinsic reward; no success broadcast',
                  q_update_order='reverse_once_after_each_rollout',terminal_bootstrap=False,
                  q_max_cache='exact_max_including_decreases',q_decay=1.,
                  intrinsic_definition='weight*positive_actual_Q_change_else_boring_penalty + visit_weight*(1/33-softmax(N)[action])',
                  count_total=cfg.initial_count*NUM_ACTIONS,count_dtype='float64',
                  count_update='increment_and_normalize_per_transition_in_forward_order',
                  pretrain_steps=0,training_cases=TRAIN_CASES,heldout_cases=HELDOUT_CASES,
                  dataset_sha256=digest,schedule=schedule,schedule_seed=cfg.seed+1,
                  initialization_seed=cfg.seed+4,gradient_clip=1,baseline=None,
                  policy_task_credit='success_and_early_bonus_every_successful_action; offending_violation_only',
                  resume_source=str(args.resume.resolve()) if args.resume else None,
                  resume_source_sha256=source_hash)
    print(json.dumps({k:v for k,v in metadata.items() if k!='schedule'}),flush=True)
    started=time.perf_counter()
    with (args.output/(prefix+'.metrics.jsonl')).open('x',encoding='utf-8',buffering=1) as stream:
        for episode in range(completed,cfg.episodes):
            metrics=train_episode(model,optimizer,memory,cfg,episode,rng,init_rng,int(schedule[episode]))
            elapsed=elapsed_before+time.perf_counter()-started
            metrics['elapsed_seconds']=elapsed
            stream.write(json.dumps(metrics,allow_nan=False)+'\n')
            if episode==completed or (episode+1)%cfg.log_every==0:
                print(f"episode {episode+1}/{cfg.episodes} | success {metrics['success_rate']:.3f} | Q rows {len(memory.q)} | seconds {elapsed:.1f}",flush=True)
            if (episode+1)%cfg.checkpoint_every==0 or episode+1==cfg.episodes:
                atomic_save(checkpoint(metadata,model,optimizer,memory,rng,init_rng,episode+1,elapsed),
                                 args.output/(prefix+'.checkpoint.pt'))
    if args.resume:
        assert sha256(args.resume)==source_hash
    atomic_save(dict(**metadata,completed_episodes=cfg.episodes,model=model.state_dict()),
                     args.output/(prefix+'.model.pt'))
    print('Finished.',flush=True)


if __name__=='__main__':
    main()
