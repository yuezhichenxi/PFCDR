"""Run the approved seed-2020 MF Book-to-Music K sweep using verified caches."""
import argparse
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime

PROJECT = Path(__file__).resolve().parents[1]
WORK = PROJECT.parent / 'ccfa-workfiles/experiments/pfcdr-book-music-k-sweep'
CACHE = WORK.parent / 'pfcdr-full-reproduction'
TRAIN_PYTHON = WORK.parent / 'pfcdr-book-movie-20/train-env/Scripts/python.exe'
PUBLIC = PROJECT / 'results/book_music_k_sweep'
KS = [10000, 15000, 20000, 25000, 30000]


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def read_json(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def write_json(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for attempt in range(20):
        try:
            temp.replace(p)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def git(*args):
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='never')
    p = subprocess.run(['git', *args], cwd=PROJECT, env=env, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       encoding='utf-8', errors='replace', timeout=90)
    return p.stdout.strip()


def publish(paths, title):
    if git('diff', '--cached', '--name-only'):
        raise RuntimeError('Unrelated staged changes; refusing to mix commits')
    git('add', '--', *paths)
    staged = git('diff', '--cached', '--name-only')
    if staged:
        message = WORK / 'commit-message.txt'
        message.write_text(title+'\n\nMF Book-to-Music, seed=2020, original training batches, '
                           'no accumulation, evaluation batch=256, only five requested K values.\n',
                           encoding='utf-8')
        git('commit', '--file', str(message))
    commit = git('rev-parse', 'HEAD')
    try:
        detail = git('-c', 'credential.interactive=false', 'push', 'origin', 'main')
        write_json(WORK / 'push-status.json', dict(ok=True, commit=commit, detail=detail, updated_at=now()))
    except Exception as error:
        write_json(WORK / 'push-status.json', dict(ok=False, commit=commit, detail=str(error), updated_at=now()))
    return commit


def predict_ks(model, x, ks):
    """Use original MF selection/voting once, then average each requested prefix."""
    import torch
    captured = {}
    def save_users(module, inputs, output):
        captured['users'] = output
    def save_items(module, inputs, output):
        captured['items'] = output
    handles = [model.tgt_model.uid_embedding.register_forward_hook(save_users),
               model.tgt_model.iid_embedding.register_forward_hook(save_items)]
    try:
        largest = max(ks)
        original = model(x, 'test_source_free', topk=largest)
    finally:
        for handle in handles:
            handle.remove()
    users = captured['users'].view(len(x), largest, model.emb_dim)
    votes = torch.sum(users * captured['items'], dim=2)
    result = {k: votes[:, :k].mean(dim=1) for k in ks}
    torch.testing.assert_close(result[largest], original, atol=5e-6, rtol=1e-6)
    return result


def worker(beta):
    import numpy as np
    import pandas as pd
    import torch
    sys.path.insert(0, str(PROJECT))
    from run import Run
    root = WORK / f'task2_beta{beta}_seed2020'
    cache = CACHE / 'runs' / f'task2_beta{beta}_seed2020' / 'attempt_1'
    spec, old = read_json(cache/'run-spec.json'), read_json(cache/'result.json')
    cfg = read_json(root/'config.json')
    manifest = read_json(PROJECT / old['dataset_manifest'])
    ratio = f"{int(cfg['ratio'][0]*10)}_{int(cfg['ratio'][1]*10)}"
    pair_name = 'tgt_CDs_and_Vinyl_src_Books'
    data_root = PROJECT/'data/ready'/f'_{ratio}'/pair_name
    proto_root = cache/'syn_data'/ratio/pair_name
    for name, expected in manifest['file_sha256'].items():
        assert sha(data_root/name) == expected, name
    test_ids = set(pd.read_csv(data_root/'test_list.csv', header=None).to_numpy().ravel().astype(int))
    train_ids = set(range(cfg['src_tgt_pairs']['2']['co_user_num']))-test_ids
    prototype_sha = {}
    for name, ids in [('5_MF_train_src.csv',train_ids), ('5_MF_train_tgt.csv',train_ids),
                      ('5_MF_test_src.csv',test_ids)]:
        values = np.loadtxt(proto_root/name, delimiter=',', ndmin=2)
        assert values.shape == (len(ids),11) and np.isfinite(values).all()
        assert set(values[:,0].astype(int)) == ids and len(np.unique(values[:,0])) == len(ids)
        prototype_sha[name] = sha(proto_root/name)
    random.seed(2020); np.random.seed(2020); torch.manual_seed(2020); torch.cuda.manual_seed(2020)
    runner = Run(cfg)
    # Recreate the original initialization and the 50 pretraining DataLoader
    # iterations (two global RNG draws each). Cached inversion uses NumPy only.
    initialized = runner.get_model()
    model = torch.load(cache/spec['baseline_checkpoint'], map_location='cpu', weights_only=False).cuda()
    assert torch.equal(initialized.rp_mapping.weight, model.rp_mapping.weight)
    del initialized
    gc.collect(); torch.cuda.empty_cache()
    for _ in range(10*cfg['epoch']):
        torch.empty((), dtype=torch.int64).random_()
    train = runner.read_rating_preference(str(proto_root/'5_MF'))
    test = runner.read_meta_test(str(proto_root/'5_MF'), str(data_root/'test.csv'))
    for p in model.parameters():
        p.requires_grad = True
    frozen = {k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('rp_mapping.')}
    optimizer = torch.optim.Adam(model.rp_mapping.parameters(), lr=cfg['lr_prototype'], weight_decay=cfg['wd'])
    criterion = torch.nn.MSELoss()
    history = {str(k):[] for k in KS}
    started = now()
    first_batch_checked = False
    for epoch in range(cfg['epoch']):
        write_json(root/'progress.json', dict(stage='training_prototype_mapping', beta=beta, epoch=epoch+1,
                                              completed_epochs=epoch, updated_at=now()))
        runner.train(train, model, criterion, optimizer, epoch, 'train_source_free', mapping=True,
                     accumulation_steps=1)
        model.eval()
        all_y, all_pred = [], {k:[] for k in KS}
        write_json(root/'progress.json', dict(stage='evaluating_five_k', beta=beta, epoch=epoch+1,
                                              completed_epochs=epoch, updated_at=now()))
        with torch.no_grad():
            for x,y in test:
                predictions = predict_ks(model, x, KS)
                if not first_batch_checked:
                    for k in KS:
                        torch.testing.assert_close(predictions[k], model(x,'test_source_free',topk=k),
                                                   atol=5e-5, rtol=1e-5)
                    first_batch_checked = True
                all_y.append(y.squeeze(1).cpu())
                for k in KS:
                    all_pred[k].append(predictions[k].cpu())
        target = torch.cat(all_y).float()
        for k in KS:
            prediction = torch.cat(all_pred[k])
            mae = torch.nn.functional.l1_loss(prediction,target).item()
            rmse = torch.nn.functional.mse_loss(prediction,target).sqrt().item()
            history[str(k)].append(dict(epoch=epoch+1,mae=mae,rmse=rmse))
            print(f'beta={beta} epoch={epoch+1} K={k} MAE={mae} RMSE={rmse}', flush=True)
        write_json(root/'partial-results.json',dict(beta=beta,seed=2020,history=history,updated_at=now()))
    for name,tensor in model.state_dict().items():
        if name in frozen:
            assert torch.equal(tensor.cpu(),frozen[name]), name
    torch.save(model, root/'model_after_prototype_mapping.pt')
    best = {k:dict(mae=min(x['mae'] for x in h),rmse=min(x['rmse'] for x in h),
                   mae_epoch=min(h,key=lambda x:x['mae'])['epoch'],
                   rmse_epoch=min(h,key=lambda x:x['rmse'])['epoch']) for k,h in history.items()}
    report = dict(complete=True,task='Book to Music',beta=beta,seed=2020,epoch=10,
                  k_values=KS,history=history,best=best,config=cfg,started_at=started,completed_at=now(),
                  cached_baseline_job=old['job_id'],cached_baseline_commit=old['execution_commit'],
                  cached_baseline_sha256=sha(cache/spec['baseline_checkpoint']),prototype_sha256=prototype_sha,
                  reused_source_target_pretraining=True,reused_inversion=True,
                  original_pytorch_rng_replayed=True,only_rp_mapping_weights_changed=True,
                  shared_k_predictions_checked_against_original_forward=True,
                  baseline_reference_from_cached_seed2020={k:v for k,v in old['metrics'].items()
                                                           if k.startswith(('tgt_','emcdr_'))},
                  aggregation='Single seed; independent minimum MAE/RMSE over epochs. No averaging.')
    write_json(root/'result.json',report)
    write_json(root/'progress.json',dict(stage='completed',beta=beta,completed_epochs=10,updated_at=now()))


def summary():
    rows = []
    for beta in [20,50,80]:
        p = PUBLIC/'trials'/f'task2_beta{beta}_seed2020.json'
        if p.exists():
            report = read_json(p)
            for k in KS:
                rows.append(dict(beta=beta,k=k,**report['best'][str(k)]))
    write_json(PUBLIC/'summary.json',dict(task='Book to Music',seed=2020,
                                         complete=len(rows)==15,completed_conditions=len(rows),
                                         planned_conditions=15,rows=rows,updated_at=now()))
    lines = ['<!-- BOOK_MUSIC_K_SWEEP_START -->',
             '| 测试比例 | K | seed | MAE | RMSE | MAE 最优 epoch | RMSE 最优 epoch |',
             '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['beta']}% | {r['k']} | 2020 | {r['mae']:.4f} | {r['rmse']:.4f} | {r['mae_epoch']} | {r['rmse_epoch']} |")
    lines.append('<!-- BOOK_MUSIC_K_SWEEP_END -->')
    readme = (PROJECT/'README.md').read_text(encoding='utf-8-sig')
    block = '\n'.join(lines)
    if '<!-- BOOK_MUSIC_K_SWEEP_START -->' in readme:
        readme = re.sub(r'<!-- BOOK_MUSIC_K_SWEEP_START -->.*?<!-- BOOK_MUSIC_K_SWEEP_END -->',
                        lambda match:block,readme,flags=re.S)
    else:
        readme += '\n\n### 图书到音乐 K 扫描实际结果（seed=2020，单次运行）\n\n'+block+'\n'
    (PROJECT/'README.md').write_text(readme,encoding='utf-8')
    return len(rows)//5


def run_queue():
    import msvcrt
    WORK.mkdir(parents=True,exist_ok=True)
    lock = (WORK/'queue.lock').open('a+b')
    lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:
        raise RuntimeError('K-sweep queue already running; no duplicate worker launched')
    assert read_json(PROJECT/'experiments/book_music_k_sweep_plan.json')['evaluation_batch_size']==256
    for beta in [20,50,80]:
        root = WORK/f'task2_beta{beta}_seed2020'
        root.mkdir(parents=True,exist_ok=True)
        public_result = PUBLIC/'trials'/f'task2_beta{beta}_seed2020.json'
        if public_result.exists():
            continue
        cache = CACHE/'runs'/f'task2_beta{beta}_seed2020'/'attempt_1'
        cfg = read_json(cache/'config.json')
        cfg['src_tgt_pairs']['2']['batchsize_test']=256
        cfg['accumulation_steps']={}
        assert cfg['task']=='2' and cfg['base_model']=='MF' and cfg['epoch']==10
        pair=cfg['src_tgt_pairs']['2']
        assert [pair[k] for k in ['batchsize_src','batchsize_tgt','batchsize_map','batchsize_meta']]==[512,512,128,512]
        write_json(root/'config.json',cfg)
        public_params = PROJECT/'experiments/book_music_k_sweep_runs'/f'task2_beta{beta}_seed2020.json'
        write_json(public_params,dict(task='2',test_percentage=beta,seed=2020,k_values=KS,
                                     config=cfg,cache_job=f'task2_beta{beta}_seed2020',
                                     reuse='Verified original pretraining and inversion; retrain prototype bridge for 10 epochs.',
                                     effective_training_batches=[512,512,128,512],gradient_accumulation=False))
        commit = publish([str(public_params.relative_to(PROJECT))],f'experiment: 图书到音乐 beta{beta} seed2020 五组K参数')
        write_json(root/'execution.json',dict(commit=commit,started_at=now()))
        env=dict(os.environ,PYTHONHASHSEED='2020',PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0')
        home=next(s.split('=',1)[1].strip() for s in TRAIN_PYTHON.parent.parent.joinpath('pyvenv.cfg').read_text().splitlines() if s.startswith('home'))
        env['PATH']=str(Path(home)/'Library/bin')+';'+env['PATH']
        with (root/'train.log').open('ab',buffering=0) as log:
            child=subprocess.Popen([str(TRAIN_PYTHON),'-u',str(Path(__file__).resolve()),'worker','--beta',str(beta)],
                                   cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
            write_json(WORK/'status.json',dict(stage='running',beta=beta,worker_pid=child.pid,
                                              completed_runs=summary(),planned_runs=3,updated_at=now()))
            code=child.wait()
        if code:
            raise RuntimeError(f'beta{beta} worker failed with exit code {code}; see {root / "train.log"}')
        write_json(public_result,read_json(root/'result.json'))
        completed=summary()
        publish([str(public_result.relative_to(PROJECT)),'results/book_music_k_sweep/summary.json','README.md'],
                f'results: 图书到音乐 beta{beta} seed2020 五组K结果')
        write_json(WORK/'status.json',dict(stage='between_runs',completed_runs=completed,planned_runs=3,updated_at=now()))
    write_json(WORK/'status.json',dict(stage='completed',completed_runs=3,planned_runs=3,
                                      completed_k_conditions=15,updated_at=now()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','worker'])
    parser.add_argument('--beta',type=int,choices=[20,50,80])
    args=parser.parse_args()
    try:
        if args.action=='worker':
            if args.beta is None:
                parser.error('--beta is required for worker')
            worker(args.beta)
        else:
            run_queue()
    except Exception:
        WORK.mkdir(parents=True,exist_ok=True)
        write_json(WORK/'failure.json',dict(updated_at=now(),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
