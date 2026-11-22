import importlib.util
from pathlib import Path
import sys
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from models import MFBasedModel

spec = importlib.util.spec_from_file_location('k_sweep', PROJECT/'experiments/run_book_music_k_sweep.py')
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


class KSweepExecutionTests(unittest.TestCase):
    def test_shared_voting_matches_individual_original_forward(self):
        torch.manual_seed(123)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MFBasedModel({'uid_src':8,'iid_src':9}, {'uid_tgt':23,'iid_tgt':11},2,4,3).to(device)
        x = torch.tensor([[0,0.2,0.3,-0.1,0.7],[2,-0.5,0.1,0.4,0.3]],device=device)
        with torch.no_grad():
            predictions = sweep.predict_ks(model,x,[1,3,7,15])
            for k,prediction in predictions.items():
                torch.testing.assert_close(prediction,model(x,'test_source_free',topk=k))
        self.assertEqual(len(model.tgt_model.uid_embedding._forward_hooks),0)
        self.assertEqual(len(model.tgt_model.iid_embedding._forward_hooks),0)

    def test_pretraining_loader_rng_replay(self):
        torch.manual_seed(2020)
        state = torch.get_rng_state()
        data = TensorDataset(torch.arange(9))
        for _ in range(50):
            for _batch in DataLoader(data,batch_size=4,shuffle=True):
                pass
        expected = torch.get_rng_state()
        torch.set_rng_state(state)
        for _ in range(100):
            torch.empty((),dtype=torch.int64).random_()
        self.assertTrue(torch.equal(expected,torch.get_rng_state()))


if __name__=='__main__':
    unittest.main()
