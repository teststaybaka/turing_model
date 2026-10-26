"""Fresh tape-and-budget REINFORCE baseline on the unchanged 16-cell 

Run `python plain_rl_16_cells_baseline.py` with no arguments to reproduce
seed 20261011: 2,250 passes over 768 training cases, 16 rollouts/update,
horizon 64, success +1 plus .01 per remaining step, input violation -1,
and zero wrong-submission or timeout penalty. There are 256 held-out cases.
The six writable bits are initialized uniformly at random each rollout.
The model has two 64-unit tanh layers and uses Adam at .0003, gradient
clipping at 1, and no entropy regularization. Its inputs are the binary
tape and remaining budget. The dataset and final parameters default below.
Optional --seed 20261012 or --seed 20261013 selects the other recorded seeds.
Results go directly to log/ with prefix plain_rl_16_cells_baseline_seed<seed>.

No statistics inputs, statistics backups, surprise reward, or UCT pretraining.
The environment, split, schedule, initialization, and credit assignment match
the statistics-and-budget experiment. Existing seed results are never
overwritten; no automatic resume or evaluation.
"""
import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


K, NUM_ACTIONS, SUBMIT = 16, 33, 32
CONTEXTS = 1024
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

def schedule_for(config):
    rng = np.random.default_rng(config.seed+1)
    return np.concatenate([rng.permutation(TRAIN_CASES) for _ in range(config.passes)])

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


PREFIX = Path(__file__).stem
INPUT_SIZE = 2*K+1


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
    seed: int = 20261011
    log_every: int = 1000
    checkpoint_every: int = 1000

    @property
    def episodes(self):
        return TRAIN_CONTEXTS*self.passes


def encode_features(cells, remaining_budget):
    if not 0 < remaining_budget <= 1:
        raise ValueError('pre-action remaining budget must be in (0,1]')
    x = np.zeros(INPUT_SIZE, dtype=np.float32)
    x[2*np.arange(K)+cells] = 1
    x[-1] = remaining_budget
    return x


class Controller(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(INPUT_SIZE, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, NUM_ACTIONS))
        nn.init.normal_(self.net[-1].weight, std=.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


def initialize(cfg):
    torch.manual_seed(cfg.seed)
    model = Controller(cfg.hidden)
    return (model, torch.optim.Adam(model.parameters(), lr=cfg.learning_rate),
            np.random.default_rng(cfg.seed), np.random.default_rng(cfg.seed+4))


def collect_episode(model, cfg, episode, rng, init_rng, coverage, case_id, action_policy=None):
    xs, actions, credits = [], [], []
    successes = submissions = violations = successful_actions = repeats = 0
    success_remaining_steps = successful_actions_remaining_steps = 0
    initial_correct = successful_initial_correct = successful_initial_wrong = 0
    first_action_submissions = first_action_successes = 0
    before, block = len(coverage), set()
    for _ in range(cfg.rollouts):
        tape = Tape(case_id, init_rng)
        started_correct = bool(np.array_equal(tape.cells[PROTECTED_BITS:], tape.expected))
        initial_correct += int(started_correct)
        seen, length = set(), 0
        submitted = success = invalid = False
        with torch.inference_mode():
            for step in range(cfg.horizon):
                x = encode_features(tape.cells, (cfg.horizon-step)/cfg.horizon)
                if action_policy is None:
                    probs = torch.softmax(model(torch.from_numpy(x)), dim=-1).numpy()
                    action = min(int(np.searchsorted(np.cumsum(probs), rng.random())), NUM_ACTIONS-1)
                else:
                    action = action_policy(tape)
                coverage.add(tape.code)
                block.add(tape.code)
                pair = (tape.code, action)
                repeats += int(pair in seen)
                seen.add(pair)
                xs.append(x)
                actions.append(action)
                length += 1
                submitted, success, invalid = tape.step(action)
                if submitted or invalid:
                    break
        remaining = cfg.horizon-length
        local = [(cfg.success_bonus+cfg.early_success_multiplier*remaining)*int(success)]*length
        if invalid:
            local[-1] += cfg.input_change_penalty
        credits.extend(local)
        successes += int(success)
        submissions += int(submitted)
        violations += int(invalid)
        successful_actions += length*int(success)
        success_remaining_steps += remaining*int(success)
        successful_actions_remaining_steps += length*remaining*int(success)
        successful_initial_correct += int(success and started_correct)
        successful_initial_wrong += int(success and not started_correct)
        first_action_submissions += int(submitted and length == 1)
        first_action_successes += int(success and length == 1)
    t = len(actions)
    timeouts = cfg.rollouts-submissions-violations
    success_sum = cfg.success_bonus*successful_actions+cfg.early_success_multiplier*successful_actions_remaining_steps
    metrics = dict(episode=episode+1, case_id=case_id, operator_id=case_id//32, input_value=case_id%32,
        pass_index=episode//TRAIN_CONTEXTS+1, case_group='original',
        transitions=t, successes=successes, submissions=submissions, input_violations=violations,
        timeouts=timeouts, successful_actions=successful_actions, success_remaining_steps=success_remaining_steps,
        successful_actions_remaining_steps=successful_actions_remaining_steps,
        success_rate=successes/cfg.rollouts, timeout_rate=timeouts/cfg.rollouts,
        input_violation_rate=violations/cfg.rollouts, mean_length=t/cfg.rollouts,
        repeat_pair_visits=repeats, states_visited=len(coverage), new_states=len(coverage)-before,
        block_states_visited=len(block), new_block_states=len(block),
        initial_correct=initial_correct, successful_initial_correct=successful_initial_correct,
        successful_initial_wrong=successful_initial_wrong,
        first_action_submissions=first_action_submissions, first_action_successes=first_action_successes,
        mean_action_success_credit=success_sum/t,
        mean_action_base_success_credit=cfg.success_bonus*successful_actions/t,
        mean_action_early_success_credit=cfg.early_success_multiplier*successful_actions_remaining_steps/t,
        mean_action_penalty_credit=cfg.input_change_penalty*violations/t,
        mean_action_wrong_submission_credit=0., mean_action_intrinsic_credit=0.,
        mean_action_credit=float(np.mean(credits)),
        mean_rollout_reward=(cfg.success_bonus*successes+cfg.early_success_multiplier*success_remaining_steps+
                             cfg.input_change_penalty*violations)/cfg.rollouts)
    return dict(features=torch.from_numpy(np.stack(xs)), actions=torch.tensor(actions,dtype=torch.long),
                credits=torch.tensor(credits,dtype=torch.float32)), metrics, block


def train_episode(model, optimizer, cfg, episode, rng, init_rng, coverage, case_id):
    batch, metrics, block = collect_episode(model,cfg,episode,rng,init_rng,coverage,case_id)
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch['features'])
    loss = reinforce_loss(logits,batch['actions'],batch['credits'],cfg.rollouts)
    if not torch.isfinite(loss):
        raise RuntimeError('nonfinite RL loss')
    loss.backward()
    norm = nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad():
        logps = F.log_softmax(logits,dim=-1)
        entropy = float(-(logps.exp()*logps).sum(-1).mean())
    metrics.update(loss=float(loss.detach()),gradient_norm=float(norm),entropy=entropy,
                   mean_action_advantage=float(batch['credits'].mean()),
                   negative_action_advantage_fraction=float((batch['credits']<0).float().mean()))
    return metrics, block


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in asdict(Config()).items():
        parser.add_argument('--'+name.replace('_','-'),type=type(default),default=default)
    parser.add_argument('--dataset',type=Path,default=DEFAULT_DATASET)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'log',
                        help='Output directory; defaults to log/ with seed-specific filenames')
    args = parser.parse_args(argv)
    cfg = Config(**{k:getattr(args,k) for k in asdict(Config())})
    if min(cfg.passes,cfg.rollouts,cfg.horizon,cfg.hidden,cfg.log_every,cfg.checkpoint_every)<=0 or cfg.seed<0:
        parser.error('invalid budget, dimensions, or seed')
    scales=[cfg.learning_rate,cfg.success_bonus,cfg.early_success_multiplier,cfg.input_change_penalty]
    if (not np.isfinite(scales).all() or cfg.learning_rate<=0 or cfg.success_bonus<0 or
            cfg.early_success_multiplier<0 or cfg.input_change_penalty>0):
        parser.error('invalid reward or learning-rate configuration')
    prefix=f'{PREFIX}_seed{cfg.seed}'
    allowed={f'{prefix}.stdout.log',f'{prefix}.stderr.log'}
    if args.output.exists() and any(p.name not in allowed or p.stat().st_size
                                   for p in args.output.glob(f'{prefix}.*')):
        parser.error('results for this seed already exist; refusing to overwrite previous results')
    return args,cfg


def main():
    args,cfg=parse_args()
    prefix=f'{PREFIX}_seed{cfg.seed}'
    _, digest = load_dataset(args.dataset)
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    model,optimizer,rng,init_rng=initialize(cfg)
    schedule=schedule_for(cfg)
    coverage=set()
    metadata=dict(config=asdict(cfg),episodes=cfg.episodes,input_mode='tape_budget',input_size=INPUT_SIZE,
        parameters=sum(p.numel() for p in model.parameters()),raw_tape_input=True,remaining_budget_input=True,
        feature_definition='32 binary-cell one-hot entries followed by (horizon-actions_already_taken)/horizon',
        actions=NUM_ACTIONS,tape_cells=K,tape_symbols=2,protected_cells=list(PROTECTED),
        dataset_sha256=digest,dataset_path=str(args.dataset.resolve()),contexts=CONTEXTS,
        training_contexts=TRAIN_CONTEXTS,heldout_contexts=len(HELDOUT_CASES),
        training_cases=TRAIN_CASES,heldout_cases=HELDOUT_CASES,split_seed=SPLIT_SEED,
        schedule_seed=cfg.seed+1,initialization_seed=cfg.seed+4,
        schedule_first_pass=schedule[:TRAIN_CONTEXTS].tolist(),schedule_last_case=int(schedule[-1]),
        initialization='uniform_random_six_answer_bits_each_rollout',statistics=False,surprise_weight=0.,
        pretraining=None,baseline=None,wrong_submission_penalty=0.,timeout_penalty=0.,
        answer_verification='SUBMIT_only',violation_write_applied=False,
        success_reward_definition='success_bonus+early_success_multiplier*(horizon-rollout_length_including_SUBMIT)',
        success_credit='every_successful_rollout_action',penalty_credit='offending_action_only',
        entropy_regularization=0,gradient_clip=1,loss_normalization='rollouts_not_actions',
        evaluation='held-out cases excluded; no automatic evaluation')
    print(json.dumps(metadata),flush=True)
    started=time.perf_counter()
    with (args.output/f'{prefix}.metrics.jsonl').open('x',encoding='utf-8',buffering=1) as stream:
        for episode,case in enumerate(schedule):
            case=int(case)
            metrics,block=train_episode(model,optimizer,cfg,episode,rng,init_rng,coverage,case)
            elapsed=time.perf_counter()-started
            metrics['elapsed_seconds']=elapsed
            stream.write(json.dumps(metrics,allow_nan=False)+'\n')
            if episode==0 or (episode+1)%cfg.log_every==0:
                print(f"episode {episode+1}/{cfg.episodes} | case {case} | success {metrics['success_rate']:.3f} | length {metrics['mean_length']:.1f} | seconds {elapsed:.1f}",flush=True)
            if (episode+1)%cfg.checkpoint_every==0 or episode+1==cfg.episodes:
                atomic_save(dict(**metadata,completed_episodes=episode+1,model=model.state_dict(),
                    optimizer=optimizer.state_dict(),numpy_rng=rng.bit_generator.state,
                    initialization_rng=init_rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                    schedule=schedule,current_case_id=case,diagnostic_coverage=sorted(coverage),
                    diagnostic_block_coverage=sorted(block),elapsed_seconds=elapsed),
                    args.output/f'{prefix}.checkpoint.pt')
    atomic_save(dict(**metadata,completed_episodes=cfg.episodes,model=model.state_dict(),
                           schedule=schedule,elapsed_seconds=time.perf_counter()-started),args.output/f'{prefix}.model.pt')
    print('Finished.',flush=True)


if __name__=='__main__':
    main()
