"""Measure inference memory only. No optimizer, backward, or parameter updates."""
import argparse
import gc
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def digest(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--ks', type=int, nargs='+', default=[10000, 15000, 20000, 25000, 30000])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required for this memory probe')
    if args.batch_size <= 0:
        parser.error('--batch-size must be positive')
    cfg = json.loads((args.job_root / 'config.json').read_text(encoding='utf-8-sig'))
    spec = json.loads((args.job_root / 'run-spec.json').read_text(encoding='utf-8-sig'))
    assert cfg['task'] == '2' and cfg['base_model'] == 'MF'
    pair = cfg['src_tgt_pairs']['2']
    assert all(1 <= k <= pair['uid_tgt'] for k in args.ks)
    ratio = '_'.join(str(int(v * 10)) for v in cfg['ratio'])
    pair_name = f"tgt_{pair['tgt']}_src_{pair['src']}"
    data = PROJECT / 'data' / 'ready' / f'_{ratio}' / pair_name
    prototype = args.job_root / 'syn_data' / ratio / pair_name / '5_MF_test_src.csv'
    test = pd.read_csv(data / 'test.csv', header=None, names=['uid', 'iid', 'y'],
                       nrows=args.batch_size)
    preferences = pd.read_csv(prototype, header=None,
                               names=['uid'] + [str(i) for i in range(cfg['emb_dim'])])
    frame = test.merge(preferences, on='uid', how='left', validate='many_to_one', sort=False)
    assert len(frame) == args.batch_size and not frame.isna().any().any()
    x = torch.tensor(frame[['iid'] + [str(i) for i in range(cfg['emb_dim'])]].to_numpy(),
                     dtype=torch.float32, device='cuda')
    model = torch.load(args.job_root / spec['baseline_checkpoint'],
                       map_location='cpu', weights_only=False).cuda().eval()
    before = digest(model)
    rows = []
    with torch.no_grad():
        for k in args.ks:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            pred = model(x, 'test_source_free', topk=k)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            assert pred.shape == (args.batch_size,) and torch.isfinite(pred).all()
            peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
            smaller = torch.cat([model(x[j:j+128], 'test_source_free', topk=k)
                                  for j in range(0, len(x), 128)])
            max_diff = float((pred - smaller).abs().max())
            torch.testing.assert_close(pred, smaller, atol=5e-5, rtol=1e-5)
            rows.append(dict(k=k, batch_size=args.batch_size, peak_allocated_mib=peak,
                             peak_reserved_mib=reserved, seconds=elapsed,
                             max_difference_from_batch128=max_diff, ok=True))
            print(json.dumps(rows[-1]), flush=True)
            del pred, smaller
    assert digest(model) == before
    report = dict(created_at=datetime.now().astimezone().isoformat(timespec='seconds'),
                  task='Book to Music', base_model='MF', mode='inference_memory_probe',
                  checkpoint_job=spec['job']['job_id'],
                  checkpoint_note='Baseline checkpoint; prototype bridge is not retrained. No accuracy claim.',
                  cuda_device=torch.cuda.get_device_name(0),
                  gpu_total_mib=torch.cuda.get_device_properties(0).total_memory/(1024**2),
                  backward_calls=0, optimizer_steps=0, model_weights_unchanged=True,
                  source_sha256={name:hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()
                                 for name in ['models.py','run.py']}, probes=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
