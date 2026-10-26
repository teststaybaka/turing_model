"""Check the plain baseline against hand-specified rewards and the existing environment."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
import uct_inspired_1024_cases_16_cells_train as suite
import plain_rl_16_cells_baseline as p


class FixedInit:
    def __init__(self,value):self.value=value
    def integers(self,*args,**kwargs):return suite.bits(self.value,6)


class PlainSuite16Tests(unittest.TestCase):
    def test_reward_assignment_and_observations(self):
        torch.set_num_threads(1)
        cfg=p.Config(rollouts=1)
        model,opt,rng,init=p.initialize(cfg)
        self.assertEqual(sum(v.numel() for v in model.parameters()),8481)
        scenarios=[('immediate',0,[32],1.63),('early',63,[20,22,24,26,28,30,32],1.57),
                   ('five_left',63,[20,22,24,26,28,30]+[0]*52+[32],1.05),
                   ('last',63,[20,22,24,26,28,30]+[0]*57+[32],1.),
                   ('wrong',63,[32],0.),('timeout',63,[0]*64,0.),('violation',63,[0,1],-1.)]
        for name,initial,actions,reward in scenarios:
            with self.subTest(name=name):
                seq=iter(actions)
                batch,m,_=p.collect_episode(model,cfg,0,rng,FixedInit(initial),set(),0,lambda _:next(seq))
                expected=np.full(len(actions),reward,dtype=np.float32)
                if name=='violation':expected[:-1]=0
                np.testing.assert_allclose(batch['credits'],expected)
                self.assertAlmostEqual(m['mean_rollout_reward'],reward)
                self.assertEqual(batch['features'].shape,(len(actions),33))
                np.testing.assert_array_equal(batch['features'][:,-1],(64-np.arange(len(actions)))/64)
                onehot=batch['features'][:,:32].numpy().reshape(-1,16,2)
                np.testing.assert_array_equal(onehot.sum(-1),1)
                np.testing.assert_array_equal(onehot.argmax(-1)[:,:10],0)
                np.testing.assert_array_equal(onehot.argmax(-1)[0,10:],suite.bits(initial,6))
                self.assertEqual(m['mean_action_intrinsic_credit'],0.)
                self.assertEqual(m['mean_action_wrong_submission_credit'],0.)
                self.assertEqual(m['submissions']+m['timeouts']+m['input_violations'],1)
                # Same path must receive exactly the existing collector's task credits with surprise disabled.
                other=suite.Config(input_mode='statistics_budget',early_success_multiplier=.01,
                                   rollouts=1,pretrain_steps=0,surprise_weight=0.)
                seq=iter(actions)
                old,oldm,_=suite.collect_episode(model,suite.Statistics(.99,100),other,0,rng,
                                                 FixedInit(initial),set(),0,lambda _:next(seq))
                np.testing.assert_array_equal(batch['credits'],old['credits'])
                for k in ('transitions','successes','successful_actions','input_violations','timeouts',
                          'first_action_submissions','successful_initial_wrong','mean_action_success_credit'):
                    self.assertEqual(m[k],oldm[k])

    def test_plain_update_uses_neither_statistics_nor_pretraining(self):
        torch.set_num_threads(1)
        with patch.object(suite,'Statistics',side_effect=AssertionError('statistics created')), \
             patch.object(suite,'path_probability_change',side_effect=AssertionError('surprise computed')), \
             patch.object(suite,'pretrain',side_effect=AssertionError('pretraining used')):
            cfg=p.Config(rollouts=4)
            model,opt,rng,init=p.initialize(cfg)
            self.assertFalse(opt.state)
            before=[v.detach().clone() for v in model.parameters()]
            m,_=p.train_episode(model,opt,cfg,0,rng,init,set(),suite.TRAIN_CASES[0])
            self.assertTrue(all(v['step'].item()==1 for v in opt.state.values()))
            self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,model.parameters())))
            self.assertTrue(all(np.isfinite(v) for v in m.values() if isinstance(v,(int,float))))

    def test_split_schedule_and_initialization_reproducible(self):
        _,digest=p.load_dataset(p.DEFAULT_DATASET)
        self.assertEqual(digest,'2b51f4e71d9ecd85a896f939c4ffc107d9902ced1ba6373503eff411decc6968')
        for seed in (20261011,20261012,20261013):
            cfg=p.Config(seed=seed,passes=2)
            schedule=p.schedule_for(cfg)
            np.testing.assert_array_equal(np.sort(schedule.reshape(2,768),axis=1),np.tile(suite.TRAIN_CASES,(2,1)))
            self.assertFalse(set(schedule)&set(suite.HELDOUT_CASES))
            same=suite.schedule_for(suite.Config(seed=seed,passes=2))
            np.testing.assert_array_equal(schedule,same)
            first=np.random.default_rng(seed+4);second=np.random.default_rng(seed+4)
            observed=set()
            for _ in range(200):
                t=p.Tape(int(schedule[0]),first)
                np.testing.assert_array_equal(t.cells[10:],second.integers(0,2,size=6))
                observed.add(t.code%64)
            self.assertGreater(len(observed),50)


if __name__=='__main__':unittest.main()
