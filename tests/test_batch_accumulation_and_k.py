"""Small correctness checks; no Amazon data or experiment checkpoints are trained."""
import copy
import math
import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import MFBasedModel
from run import Run


class SmallModel(torch.nn.Module):
    def __init__(self, mapping=False):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3 if mapping else 1)
        self.mapping = mapping

    def forward(self, x, stage):
        out = self.linear(x)
        return (out, x * 0.7 + 0.2) if self.mapping else out.squeeze(1)


class BatchAccumulationTests(unittest.TestCase):
    def compare(self, n, microbatch, mapping=False, use_config=False):
        torch.manual_seed(41)
        accumulated = SmallModel(mapping)
        reference = copy.deepcopy(accumulated)
        x, y = torch.randn(n, 3), torch.randn(n, 1)
        criterion = torch.nn.MSELoss()
        a = torch.optim.Adam(accumulated.parameters(), lr=0.005, weight_decay=0.0001)
        b = torch.optim.Adam(reference.parameters(), lr=0.005, weight_decay=0.0001)
        runner = Run.__new__(Run)
        runner.accumulation_steps = {'check': 2} if use_config else {}
        runner.train(DataLoader(TensorDataset(x, y), batch_size=microbatch),
                     accumulated, criterion, a, 0, 'check', mapping=mapping,
                     accumulation_steps=None if use_config else 2)
        for xb, yb in DataLoader(TensorDataset(x, y), batch_size=microbatch * 2):
            if mapping:
                pred, target = reference(xb, 'check')
                loss = criterion(pred, target)
            else:
                loss = criterion(reference(xb, 'check'), yb.squeeze().float())
            reference.zero_grad()
            loss.backward()
            b.step()
        for left, right in zip(accumulated.parameters(), reference.parameters()):
            torch.testing.assert_close(left, right, atol=2e-7, rtol=2e-6)
            torch.testing.assert_close(a.state[left]['exp_avg'], b.state[right]['exp_avg'],
                                       atol=2e-7, rtol=2e-6)
            self.assertEqual(int(a.state[left]['step']), math.ceil(n / (microbatch * 2)))

    def test_512_twice_matches_1024(self):
        self.compare(2048, 512)

    def test_single_small_final_group(self):
        self.compare(1300, 512)

    def test_uneven_microbatches_are_sample_weighted(self):
        self.compare(800, 512)

    def test_vector_mapping_loss_and_config(self):
        self.compare(1300, 512, mapping=True, use_config=True)

    def test_default_keeps_original_per_batch_updates(self):
        torch.manual_seed(42)
        model = SmallModel()
        expected = copy.deepcopy(model)
        data = TensorDataset(torch.randn(10, 3), torch.randn(10, 1))
        opt = torch.optim.Adam(model.parameters(), lr=0.005)
        refopt = torch.optim.Adam(expected.parameters(), lr=0.005)
        runner = Run.__new__(Run)
        runner.train(DataLoader(data, batch_size=4), model, torch.nn.MSELoss(), opt, 0, 'check')
        for x, y in DataLoader(data, batch_size=4):
            loss = torch.nn.MSELoss()(expected(x, 'check'), y.squeeze().float())
            expected.zero_grad()
            loss.backward()
            refopt.step()
        for left, right in zip(model.parameters(), expected.parameters()):
            self.assertTrue(torch.equal(left, right))


class ExplicitKTests(unittest.TestCase):
    def test_voting_for_different_k(self):
        torch.manual_seed(43)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MFBasedModel({'uid_src': 5, 'iid_src': 7},
                             {'uid_tgt': 9, 'iid_tgt': 11}, 2, 3, 4).to(device)
        x = torch.tensor([[0, 0.2, -0.3, 0.7], [1, -0.1, 0.6, 0.4]], device=device)
        with torch.no_grad():
            prototype = model.rp_mapping(x[:, 1:])
            users = model.tgt_model.uid_embedding.weight
            items = model.tgt_model.iid_embedding(x[:, 0].long())
            for k in [1, 3, 9]:
                # Independently calculate each user's selected voters and average score.
                expected = []
                for row in range(len(x)):
                    errors = (users @ prototype[row] - 5).abs()
                    selected = torch.topk(errors, k, largest=False).indices
                    expected.append((users[selected] @ items[row]).mean())
                actual = model(x, 'test_source_free', topk=k)
                torch.testing.assert_close(actual, torch.stack(expected))
            for k in [0, 10, 2.5]:
                with self.assertRaises(ValueError):
                    model(x, 'test_source_free', topk=k)

    def test_default_uses_10000(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MFBasedModel({'uid_src': 3, 'iid_src': 4},
                             {'uid_tgt': 70001, 'iid_tgt': 5}, 2, 2, 10000).to(device)
        x = torch.tensor([[0, 0.2, -0.3], [1, 0.7, 0.4]], device=device)
        with torch.no_grad():
            torch.testing.assert_close(model(x, 'test_source_free'),
                                       model(x, 'test_source_free', topk=10000))


if __name__ == '__main__':
    unittest.main()
