import sys
import unittest
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
sys.path.insert(0,str(PROJECT/'experiments'))
import run_emcdr_book_music_fresh as fresh
import run_emcdr_book_music_lr_grid as grid
from models import MFBasedModel
from run import Run


class LearningRateGridTests(unittest.TestCase):
    def test_only_learning_rate_and_output_location_change(self):
        for beta in grid.BETAS:
            expected=fresh.config_for(beta)
            expected.pop('lr');expected.pop('model_root')
            for lr in grid.LRS:
                actual=grid.config_for(beta,lr)
                self.assertEqual(actual.pop('lr'),lr)
                actual.pop('model_root')
                self.assertEqual(actual,expected)

    def test_all_three_actual_optimizers_receive_candidate(self):
        model=MFBasedModel({'uid_src':3,'iid_src':5},{'uid_tgt':4,'iid_tgt':6},2,10,3)
        runner=Run.__new__(Run)
        runner.wd=0
        for lr in grid.LRS:
            runner.lr=lr
            for optimizer in runner.get_optimizer(model):
                self.assertEqual(optimizer.param_groups[0]['lr'],lr)
                self.assertEqual(optimizer.param_groups[0]['weight_decay'],0)

    def test_job_directories_are_distinct_and_grid_is_bounded(self):
        names={grid.job_id(beta,lr) for beta in grid.BETAS for lr in grid.LRS}
        self.assertEqual(len(names),15)
        with self.assertRaises(ValueError):
            grid.job_id(40,0.01)
        with self.assertRaises(ValueError):
            grid.job_id(20,0.003)


if __name__=='__main__':
    unittest.main()
