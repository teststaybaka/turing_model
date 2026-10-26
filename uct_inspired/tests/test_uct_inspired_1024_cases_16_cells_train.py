"""Environment, rewards, pretraining and exact continuation checks for the standalone learner."""
import contextlib
import copy
import io
import json
import unittest
import uuid
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import uct_inspired_1024_cases_16_cells_train as b



class FixedInit:
    def __init__(self, value): self.value=value
    def integers(self, *args, **kwargs): return b.bits(self.value,6)


class Suite16Tests(unittest.TestCase):
    def test_tape_statistics_budget_preserves_teacher_and_credits(self):
        torch.set_num_threads(1)
        oldcfg=b.Config(input_mode='statistics_budget',early_success_multiplier=.01,rollouts=1,pretrain_steps=3)
        cfg=b.Config(input_mode='tape_statistics_budget',early_success_multiplier=.01,rollouts=1,pretrain_steps=3)
        x,target=b.teacher_batch(np.random.default_rng(7),cfg)
        oldx,oldtarget=b.teacher_batch(np.random.default_rng(7),oldcfg)
        self.assertEqual(tuple(x.shape),(256,99))
        np.testing.assert_array_equal(x[:,32:],oldx)
        np.testing.assert_array_equal(target,oldtarget)
        tape_features=x[:,:32].numpy().reshape(256,16,2)
        np.testing.assert_array_equal(tape_features.sum(axis=-1),1)
        self.assertTrue(np.isin(tape_features,[0,1]).all())
        self.assertTrue((tape_features.argmax(-1).var(axis=0)>0).all())
        model,opt,stats,rng,init=b.initialize(cfg)
        self.assertEqual(sum(p.numel() for p in model.parameters()),12705)
        rng_before=(str(rng.bit_generator.state),str(init.bit_generator.state))
        result=b.pretrain(model,cfg,batch_fn=b.teacher_batch)
        self.assertEqual(result['completed_steps'],3)
        self.assertEqual(len(stats),0);self.assertFalse(opt.state)
        self.assertEqual(rng_before,(str(rng.bit_generator.state),str(init.bit_generator.state)))
        for actions in ([20,22,24,26,28,30,32],[0,1],[32],[0]*64):
            seq=iter(actions)
            batch,m,_=b.collect_episode(model,b.Statistics(.99,100),cfg,0,rng,FixedInit(63),set(),0,lambda _:next(seq))
            seq=iter(actions)
            old,oldm,_=b.collect_episode(model,b.Statistics(.99,100),oldcfg,0,rng,FixedInit(63),set(),0,lambda _:next(seq))
            np.testing.assert_array_equal(batch['features'][:,32:],old['features'])
            np.testing.assert_array_equal(batch['credits'],old['credits'])
            self.assertEqual(m,oldm)
            recovered=batch['features'][:,:32].numpy().reshape(-1,16,2).argmax(-1)
            tape=b.Tape(0,FixedInit(63))
            for row,action in zip(recovered,actions):
                np.testing.assert_array_equal(row,tape.cells)
                tape.step(action)
        b.train_episode(model,opt,stats,cfg,0,rng,init,set(),b.TRAIN_CASES[0])
        self.assertTrue(all(v['step'].item()==1 for v in opt.state.values()))

    def test_oracles_and_dataset(self):
        tables=np.random.default_rng(20261022).integers(0,64,(2,32))
        for x in range(32):
            s=f'{x:05b}';a=x//4;c=x%4
            crc=0
            for char in s:
                high=bool(crc & 32) != bool(int(char))
                crc=(crc*2)%64
                if high:crc^=3
            expected=[a+c,(a-c)%64,a&c,a^c,s.count('1'),int(s[::-1],2),
                      4*int(a<c)+2*int(a==c)+int(a>c),int(''.join(sorted(s)),2),a*c,a|c,
                      int(s[1:]+s[0],2),int(s[::-1],2) if s[0]=='1' else x,
                      len([v for v in s.split('0') if v]),sum(s[i]!=s[i+1] for i in range(4)),
                      int(''.join(str(int(s[i]!=s[(i+1)%5])) for i in range(5)),2),x^(x//2),
                      31-x,x*2,x//2,int(s[-1]+s[:-1],2),int(s[1]+s[0]+s[3]+s[2]+s[4],2),
                      int(''.join(str(s[:i+1].count('1')%2) for i in range(5)),2),
                      int(''.join(str(s[i:].count('1')%2) for i in range(5)),2),
                      2**(x%6),x**2%64,(13*x+7)%64,crc,(x^21).bit_count(),
                      max(map(len,s.split('0'))),int(tables[0,x]),int(tables[1,x]),
                      int(s[2]+s[4]+s[0]+s[3]+s[1],2)]
            self.assertEqual([b.oracle(op,x) for op in range(32)],expected)
        self.assertEqual(len({tuple(b.oracle(op,x) for x in range(32)) for op in range(32)}),32)
        spec=b.dataset_spec()
        self.assertEqual(len(spec['cases']),1024)
        self.assertTrue(all(len(r['answer'])==6 for r in spec['cases']))

    def test_split_and_all_seed_schedules(self):
        train=set(b.TRAIN_CASES);held=set(b.HELDOUT_CASES)
        self.assertEqual(len(train),768);self.assertEqual(len(held),256)
        self.assertFalse(train & held);self.assertEqual(train | held,set(range(1024)))
        for op in range(32):
            self.assertEqual(sum(c//32==op for c in train),24)
            self.assertEqual(sum(c//32==op for c in held),8)
        for seed in (20261011,20261012,20261013):
            cfg=b.Config(seed=seed)
            sch=b.schedule_for(cfg).reshape(cfg.passes,768)
            np.testing.assert_array_equal(np.sort(sch,axis=1),np.tile(b.TRAIN_CASES,(2250,1)))
            self.assertEqual(cfg.episodes,1728000)

    def test_all_answers_and_protected_writes(self):
        for case in range(1024):
            target=b.oracle(case//32,case%32)
            for answer in range(64):
                t=b.Tape(case,FixedInit(answer))
                self.assertEqual(t.code,case*64+answer)
                self.assertEqual(t.step(b.SUBMIT),(True,answer==target,False))
            for action in range(20):
                t=b.Tape(case,FixedInit(23));before=t.cells.copy();code=t.code
                self.assertEqual(t.step(action),(False,False,action%2!=before[action//2]))
                np.testing.assert_array_equal(t.cells,before);self.assertEqual(t.code,code)
            for answer in (0,21,42,63):
                for action in range(20,32):
                    t=b.Tape(case,FixedInit(answer));expected=t.cells.copy()
                    expected[action//2]=action%2
                    self.assertEqual(t.step(action),(False,False,False))
                    np.testing.assert_array_equal(t.cells,expected)
                    self.assertEqual(t.code,int(''.join(map(str,expected)),2))

    def test_pretraining_features_and_optimizer(self):
        torch.set_num_threads(1)
        cfg=b.Config(input_mode='statistics_only',early_success_multiplier=0.,pretrain_steps=3)
        model,opt,stats,rng,init=b.initialize(cfg)
        x,p=b.teacher_batch(np.random.default_rng(7),cfg)
        self.assertEqual(tuple(x.shape),(256,66));self.assertEqual(p.shape,(256,33))
        np.testing.assert_allclose(p.sum(1),1)
        n=np.full((1,33),100.);w=np.zeros_like(n)
        np.testing.assert_array_equal(b.encode_features(None,n,w,'statistics_only'),
                                      b.encode_features(np.ones((1,16)),n,w,'statistics_only'))
        self.assertEqual(sum(v.numel() for v in model.parameters()),10593)
        before=(str(rng.bit_generator.state),str(init.bit_generator.state))
        oldstep=b.Tape.step
        def forbidden(*args):raise AssertionError('pretraining touched environment')
        b.Tape.step=forbidden
        try:result=b.pretrain(model,cfg,batch_fn=b.teacher_batch)
        finally:b.Tape.step=oldstep
        self.assertEqual(result['completed_steps'],3);self.assertFalse(opt.state)
        self.assertEqual(len(stats),0)
        self.assertEqual(before,(str(rng.bit_generator.state),str(init.bit_generator.state)))
        b.train_episode(model,opt,stats,cfg,0,rng,init,set(),b.TRAIN_CASES[0])
        self.assertTrue(all(v['step'].item()==1 for v in opt.state.values()))

    def test_credits_counts_and_clock(self):
        cfg=b.Config(input_mode='statistics_only',early_success_multiplier=0.,rollouts=1,pretrain_steps=0)
        for kind,actions in [('success',[20,22,24,26,28,30,32]),('violation',[0,1]),
                             ('timeout',[21]*64),('wrong',[32])]:
            model,opt,stats,rng,init=b.initialize(cfg);seq=iter(actions)
            batch,m,_=b.collect_episode(model,stats,cfg,0,rng,FixedInit(63),set(),0,lambda t:next(seq))
            self.assertEqual(batch['features'].shape[1],66)
            self.assertEqual(m['successes'],kind=='success')
            self.assertEqual(m['input_violations'],kind=='violation')
            self.assertEqual(m['timeouts'],kind=='timeout')
            self.assertEqual(m['mean_action_wrong_submission_credit'],0.)
            expected=m['mean_action_success_credit']+m['mean_action_penalty_credit']+m['mean_action_intrinsic_credit']
            self.assertAlmostEqual(float(batch['credits'].mean()),expected,places=6)
            snap=stats.snapshot()
            self.assertEqual(float(snap['values'][:,0].sum()),len(stats)*33*100+len(actions))
            self.assertEqual(float(snap['values'][:,1].sum()),len(stats)*33*50+m['successful_actions'])
            self.assertEqual(float(snap['failure_counts'].sum()),len(stats)*33*50+m['input_violations'])
            key=int(snap['keys'][0]);before=stats.values(key).copy()
            stats.row(64,20);np.testing.assert_array_equal(stats.values(key),before)
            stats.row(key,2);np.testing.assert_allclose(stats.values(key),before*.99**2,rtol=1e-6)

    def test_remaining_budget_and_early_success_credit(self):
        torch.set_num_threads(1)
        cfg=b.Config(input_mode='statistics_budget',early_success_multiplier=.01,
                     rollouts=1,pretrain_steps=3,surprise_weight=0.)
        model,opt,stats,rng,init=b.initialize(cfg)
        self.assertEqual(sum(p.numel() for p in model.parameters()),10657)
        x,target=b.teacher_batch(np.random.default_rng(7),cfg)
        oldx,oldtarget=b.teacher_batch(np.random.default_rng(7),b.Config(input_mode='statistics_only',early_success_multiplier=0.))
        self.assertEqual(tuple(x.shape),(256,67))
        np.testing.assert_array_equal(x[:,:66],oldx)
        np.testing.assert_array_equal(target,oldtarget)
        self.assertTrue(((x[:,-1]>=1/64)&(x[:,-1]<=1)).all())
        np.testing.assert_array_equal(x[:,-1]*64,np.round(x[:,-1]*64))
        self.assertGreater(len(np.unique(x[:,-1])),1)
        b.pretrain(model,cfg,batch_fn=b.teacher_batch)
        self.assertFalse(opt.state);self.assertEqual(len(stats),0)
        scenarios=[('immediate',0,[32],1.63),
                   ('early',63,[20,22,24,26,28,30,32],1.57),
                   ('five_left',63,[20,22,24,26,28,30]+[0]*52+[32],1.05),
                   ('last',63,[20,22,24,26,28,30]+[0]*57+[32],1.),
                   ('wrong',63,[32],0.),('timeout',63,[0]*64,0.),('violation',63,[0,1],-1.)]
        for name,initial,actions,reward in scenarios:
            s=b.Statistics(.99,100.);seq=iter(actions)
            batch,m,_=b.collect_episode(model,s,cfg,0,rng,FixedInit(initial),set(),0,lambda t:next(seq))
            expected=np.full(len(actions),reward,dtype=np.float32)
            if name=='violation':expected[:-1]=0
            np.testing.assert_allclose(batch['credits'],expected)
            np.testing.assert_array_equal(batch['features'][:,-1],(64-np.arange(len(actions)))/64)
            self.assertAlmostEqual(m['mean_rollout_reward'],reward)
            self.assertEqual(m['success_remaining_steps'],(64-len(actions))*m['successes'])
            self.assertEqual(m['successful_actions_remaining_steps'],len(actions)*(64-len(actions))*m['successes'])
            self.assertAlmostEqual(m['mean_action_success_credit'],m['mean_action_base_success_credit']+m['mean_action_early_success_credit'])
            snap=s.snapshot()
            self.assertEqual(float(snap['values'][:,0].sum()),len(s)*33*100+len(actions))
            self.assertEqual(float(snap['values'][:,1].sum()),len(s)*33*50+len(actions)*m['successes'])
            self.assertEqual(float(snap['failure_counts'].sum()),len(s)*33*50+m['input_violations'])
        b.train_episode(model,opt,stats,cfg,0,rng,init,set(),b.TRAIN_CASES[0])
        self.assertTrue(all(v['step'].item()==1 for v in opt.state.values()))


def fixture(input_mode='statistics_budget'):
    torch.set_num_threads(1)
    cfg = b.Config(input_mode=input_mode,rollouts=2,horizon=4,pretrain_steps=0)
    model,opt,stats,rng,init = b.initialize(cfg)
    coverage = set()
    cases = b.TRAIN_CASES[:3]
    for i,case in enumerate(cases[:2]):
        b.train_episode(model,opt,stats,cfg,i,rng,init,coverage,case)
    checkpoint = copy.deepcopy(dict(config=asdict(cfg),completed_episodes=2,
        training_cases=b.TRAIN_CASES,heldout_cases=b.HELDOUT_CASES,schedule=np.array(cases),
        model=model.state_dict(),optimizer=opt.state_dict(),statistics=stats.snapshot(),
        diagnostic_coverage=sorted(coverage),numpy_rng=rng.bit_generator.state,
        initialization_rng=init.bit_generator.state,torch_rng=torch.get_rng_state()))
    return checkpoint,(cfg,model,opt,stats,rng,init,coverage)


class ContinuationTests(unittest.TestCase):
    def test_chained_restore_and_new_only_schedule(self):
        checkpoint,_ = fixture()
        cfg,model,opt,stats,rng,init,coverage,visits = b.restore_checkpoint(checkpoint)
        case = b.HELDOUT_CASES[0]
        b.train_episode(model,opt,stats,cfg,0,rng,init,coverage,case,statistics_episode=0)
        visits[case]+=1
        chained=copy.deepcopy(dict(config=asdict(cfg),completed_episodes=3,
            original_training_cases=b.TRAIN_CASES,newly_added_cases=b.HELDOUT_CASES,
            case_updates=visits,model=model.state_dict(),optimizer=opt.state_dict(),statistics=stats.snapshot(),
            diagnostic_coverage=sorted(coverage),numpy_rng=rng.bit_generator.state,
            initialization_rng=init.bit_generator.state,torch_rng=torch.get_rng_state()))
        rcfg,rm,ro,rs,rr,ri,rv,recovered=b.restore_checkpoint(chained)
        np.testing.assert_array_equal(recovered,visits)
        oldkeys=np.array([key for key in coverage if key//64 in b.TRAIN_CASES],dtype=np.int64)
        oldcounts=rs.counts[oldkeys].copy();oldages=rs.ages[oldkeys].copy()
        b.train_episode(rm,ro,rs,rcfg,0,rr,ri,rv,case,statistics_episode=int(recovered[case]))
        np.testing.assert_array_equal(rs.counts[oldkeys],oldcounts)
        np.testing.assert_array_equal(rs.ages[oldkeys],oldages)
        cases,schedule=b.continuation_schedule('new',512,20261022)
        self.assertEqual(len(cases),256)
        self.assertEqual(len(schedule),131072)
        self.assertFalse(set(schedule)&set(b.TRAIN_CASES))
        self.assertIsNone(b.group_summary([],'original',16))

    def test_exact_next_update_matches_uninterrupted(self):
        for mode in ('statistics_budget', 'tape_statistics_budget'):
            with self.subTest(input_mode=mode):
                self.assert_exact_next_update(mode)

    def assert_exact_next_update(self, mode):
        checkpoint, original = fixture(mode)
        restored = b.restore_checkpoint(copy.deepcopy(checkpoint))
        cfg,model,opt,stats,rng,init,coverage = original
        rcfg,rm,ro,rs,rr,ri,rv,visits = restored
        self.assertTrue(np.array_equal(rs.snapshot()['values'],stats.snapshot()['values']))
        self.assertTrue(np.array_equal(rs.snapshot()['failure_counts'],stats.snapshot()['failure_counts']))
        self.assertTrue(np.array_equal(rs.snapshot()['ages'],stats.snapshot()['ages']))
        case = b.TRAIN_CASES[2]
        b.train_episode(model,opt,stats,cfg,2,rng,init,coverage,case)
        b.train_episode(rm,ro,rs,rcfg,0,rr,ri,rv,case,statistics_episode=int(visits[case]))
        for k,v in model.state_dict().items():
            self.assertTrue(torch.equal(v,rm.state_dict()[k]))
        for a,z in zip(opt.state.values(),ro.state.values()):
            for key in a:
                self.assertTrue(torch.equal(a[key],z[key]))
        for key in ('keys','values','failure_counts','ages','birth_ages'):
            np.testing.assert_array_equal(stats.snapshot()[key],rs.snapshot()[key])
        self.assertEqual(rng.bit_generator.state,rr.bit_generator.state)
        self.assertEqual(init.bit_generator.state,ri.bit_generator.state)

    def test_old_and_new_clocks_and_count_accounting(self):
        checkpoint,_ = fixture()
        cfg,model,opt,stats,rng,init,coverage,visits = b.restore_checkpoint(checkpoint)
        totals = b.effective_totals(stats,visits)
        weights = {k:v.clone() for k,v in model.state_dict().items()}
        initial_visits=visits.copy()
        cases = [b.HELDOUT_CASES[0],b.TRAIN_CASES[0],b.HELDOUT_CASES[0]]
        seen_ages=[]
        for i,case in enumerate(cases):
            m,_=b.train_episode(model,opt,stats,cfg,i,rng,init,coverage,case,
                                 statistics_episode=int(visits[case]))
            seen_ages.append(m['statistics_episode'])
            visits[case]+=1
            totals[case]*=.99
            totals[case]+=m['new_states']*33*np.array([100,50,50])
            totals[case]+=[m['transitions'],m['successful_actions'],m['input_violations']]
        self.assertEqual(seen_ages,[0,1,1])
        self.assertEqual(visits[b.TRAIN_CASES[1]],initial_visits[b.TRAIN_CASES[1]])
        np.testing.assert_allclose(b.effective_totals(stats,visits),totals,rtol=3e-5,atol=.1)
        self.assertTrue(any(not torch.equal(v,weights[k]) for k,v in model.state_dict().items()))
        self.assertTrue(all(int(s['step'])==5 for s in opt.state.values()))


@contextlib.contextmanager
def temporary_output():
    # Python 3.14's mode-0700 temporary directories are inaccessible to the
    # Windows sandbox identity; normal inherited workspace permissions work.
    root = Path(__file__).resolve().parents[1]
    folder = root/'.cache'/('training_workflow_test_'+uuid.uuid4().hex)
    folder.mkdir(parents=True)
    try:
        yield str(folder)
    finally:
        assert folder.resolve().is_relative_to(root/'.cache')
        for file in folder.iterdir():
            file.unlink()
        folder.rmdir()


class TrainingWorkflowTests(unittest.TestCase):
    def test_defaults_and_stage_selection(self):
        with temporary_output() as tmp:
            args, cfg = b.parse_args(['--output', tmp])
            self.assertEqual(cfg.input_mode, 'tape_statistics_budget')
            self.assertEqual((cfg.pretrain_steps, cfg.pretrain_batch_size), (50000, 256))
            self.assertEqual((cfg.passes, cfg.early_success_multiplier), (2250, .01))
            args, _ = b.parse_args(['--stage', 'new', '--checkpoint', 'source.pt', '--output', tmp])
            self.assertEqual(args.passes, 512)
            self.assertEqual(b.stage_prefix(20261013, 'new', 3),
                             b.PREFIX+'_seed20261013_stage03_new')

    def test_pretrain_original_new_and_mixed_preserve_state(self):
        torch.set_num_threads(1)
        with temporary_output() as tmp, contextlib.redirect_stdout(io.StringIO()):
            output = Path(tmp)
            b.main(['--output', tmp, '--passes', '1', '--rollouts', '1', '--horizon', '2',
                    '--pretrain-steps', '2', '--pretrain-batch-size', '8'])
            first = output/(b.stage_prefix(20261011, 'original')+'.checkpoint.pt')
            original = torch.load(first, weights_only=False)
            self.assertEqual(original['completed_episodes'], 768)
            self.assertEqual(original['pretraining']['completed_steps'], 2)
            pretrained = output/(b.stage_prefix(20261011, 'pretrain')+'.model.pt')
            self.assertTrue(pretrained.is_file())
            first_hash = b.sha256(first)
            # Continuations must never call pretraining and must keep original files intact.
            with patch.object(b, 'pretrain', side_effect=AssertionError('pretraining rerun')):
                b.main(['--output', tmp, '--stage', 'new', '--checkpoint', str(first), '--passes', '1'])
                second = output/(b.stage_prefix(20261011, 'new')+'.checkpoint.pt')
                new = torch.load(second, weights_only=False)
                self.assertEqual(new['completed_episodes'], 1024)
                np.testing.assert_array_equal(new['case_updates'], np.ones(1024))
                second_hash = b.sha256(second)
                b.main(['--output', tmp, '--stage', 'mixed', '--checkpoint', str(second), '--passes', '1'])
            third = output/(b.stage_prefix(20261011, 'mixed')+'.checkpoint.pt')
            mixed = torch.load(third, weights_only=False)
            self.assertEqual(mixed['completed_episodes'], 2048)
            np.testing.assert_array_equal(mixed['case_updates'], np.full(1024, 2))
            self.assertTrue(all(int(s['step']) == 2048 for s in mixed['optimizer']['state'].values()))
            self.assertEqual(b.sha256(first), first_hash)
            self.assertEqual(b.sha256(second), second_hash)
            for stage in ('new', 'mixed'):
                summary = json.loads((output/(b.stage_prefix(20261011, stage)+'.summary.json')).read_text())
                self.assertTrue(summary['counts_reconstructed'])
            # A different completed stage must not block a new one, but reruns must fail.
            with self.assertRaises(ValueError):
                b.main(['--output', tmp, '--stage', 'mixed', '--checkpoint', str(second), '--passes', '1'])
            self.assertEqual(mixed['config'], original['config'])
            self.assertFalse(any(p.is_dir() for p in output.iterdir()))


if __name__ == '__main__':
    unittest.main()
