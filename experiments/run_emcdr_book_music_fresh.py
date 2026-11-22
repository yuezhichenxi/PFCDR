"""Fresh MF/EMCDR training; never load a pretrained checkpoint or prototypes."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_book_music_k_sweep import git, now, read_json, sha, write_json

PROJECT=Path(__file__).resolve().parents[1]
WORK=PROJECT.parent/'ccfa-workfiles/experiments/pfcdr-emcdr-book-music-fresh-20261117'
PUBLIC=PROJECT/'results/emcdr_book_music_fresh'
PLAN=PROJECT/'experiments/emcdr_book_music_fresh_plan.json'
TRAIN_PYTHON=WORK.parent/'pfcdr-book-movie-20/train-env/Scripts/python.exe'


def publish(paths,title):
    if git('diff','--cached','--name-only'):
        raise RuntimeError('Refusing to mix unrelated staged changes')
    git('add','--',*paths)
    if git('diff','--cached','--name-only'):
        message=WORK/'commit-message.txt'
        message.write_text(title+'\n\nFresh Book-to-Music MF/EMCDR, seed=2020; no checkpoint reuse; '
                           '10 epochs per stage; original batches 512/512/128, eval=256.\n',encoding='utf-8')
        git('commit','--file',str(message))
    commit=git('rev-parse','HEAD')
    try:
        detail=git('-c','credential.interactive=false','push','origin','main')
        write_json(WORK/'push-status.json',dict(ok=True,commit=commit,detail=detail,updated_at=now()))
    except Exception as error:
        write_json(WORK/'push-status.json',dict(ok=False,commit=commit,detail=str(error),updated_at=now()))
    return commit


def config_for(beta):
    # Read the author's configuration, rather than the locally adjusted eval batch.
    cfg=json.loads(git('show','c88e6fe:config.json'))
    cfg.update(task='2',base_model='MF',epoch=10,lr=0.01,lr_prototype=0.005,
               ratio=[(100-beta)/100,beta/100],root=(PROJECT/'data').as_posix()+'/',
               model_root=(WORK/f'beta{beta}_seed2020/model').as_posix()+'/',
               accumulation_steps={})
    p=cfg['src_tgt_pairs']['2']
    assert [p[k] for k in ['batchsize_src','batchsize_tgt','batchsize_map','batchsize_test']]==[512,512,128,256]
    return cfg


def worker(beta, job_root=None):
    import numpy as np
    import torch
    sys.path.insert(0,str(PROJECT))
    from run import Run
    root=Path(job_root) if job_root is not None else WORK/f'beta{beta}_seed2020'
    if any(root.rglob('*.pt')):
        raise RuntimeError('Fresh-run folder already contains model files; refusing checkpoint reuse')
    cfg=read_json(root/'config.json')
    manifest=read_json(PROJECT/f'results/full_reproduction/datasets/task2_beta{beta}.json')
    ratio=f"{int(cfg['ratio'][0]*10)}_{int(cfg['ratio'][1]*10)}"
    data=PROJECT/'data/ready'/f'_{ratio}'/'tgt_CDs_and_Vinyl_src_Books'
    for name,expected in manifest['file_sha256'].items():
        assert sha(data/name)==expected,name
    random.seed(2020);np.random.seed(2020);torch.manual_seed(2020);torch.cuda.manual_seed(2020)

    class RecordedRun(Run):
        def __init__(self,config):
            super().__init__(config)
            self.history={'tgt':[],'emcdr':[]}

        def train(self,*args,**kwargs):
            stage=kwargs.get('stage',args[5] if len(args)>5 else None)
            epoch=kwargs.get('epoch',args[4] if len(args)>4 else None)
            write_json(root/'progress.json',dict(stage=stage,epoch=epoch+1,beta=beta,updated_at=now()))
            return super().train(*args,**kwargs)

        def eval_mae(self,model,loader,stage):
            method={'test_tgt':'tgt','test_map':'emcdr'}[stage]
            write_json(root/'progress.json',dict(stage=stage,epoch=len(self.history[method])+1,
                                                beta=beta,updated_at=now()))
            mae,rmse=super().eval_mae(model,loader,stage)
            self.history[method].append(dict(epoch=len(self.history[method])+1,mae=mae,rmse=rmse))
            write_json(root/'partial-results.json',dict(beta=beta,seed=2020,history=self.history,updated_at=now()))
            if method=='emcdr':
                for metric,value in [('mae',mae),('rmse',rmse)]:
                    if value==min(v[metric] for v in self.history[method]):
                        torch.save(model.mapping.state_dict(),root/f'mapping_best_{metric}.pt')
            return mae,rmse

    runner=RecordedRun(cfg)
    started=now()
    # Identical fresh branch ordering to Run.main(), stopping before PFCDR.
    model=runner.get_model()
    assert isinstance(model.mapping,torch.nn.Linear) and model.mapping.bias is None
    assert sum(p.numel() for p in model.mapping.parameters())==100
    initial_mapping_sha=__import__('hashlib').sha256(model.mapping.weight.detach().cpu().numpy().tobytes()).hexdigest()
    data_src,data_tgt,data_test,data_map,test_idx,mapping_idx=runner.get_data(runner.co_user_num)
    optimizer_src,optimizer_tgt,optimizer_map=runner.get_optimizer(model)
    criterion=torch.nn.MSELoss()
    runner.TgtOnly(model,data_tgt,data_test,criterion,optimizer_tgt)
    runner.EMCDR(model,data_src,data_map,data_test,criterion,optimizer_src,optimizer_map)
    root.joinpath('model').mkdir(exist_ok=True)
    torch.save(model,root/'model/final_emcdr.pt')
    metrics={k:v for k,v in runner.results.items() if k.startswith(('tgt_','emcdr_'))}
    for method,history in runner.history.items():
        assert len(history)==10
        for metric in ['mae','rmse']:
            assert metrics[f'{method}_{metric}']==min(x[metric] for x in history)
    report=dict(complete=True,task='Book to Music',method='EMCDR',base_model='MF',beta=beta,seed=2020,
                started_at=started,completed_at=now(),metrics=metrics,history=runner.history,config=cfg,
                checkpoint_loaded=False,source_and_target_trained_from_scratch=True,
                inversion_executed=False,pfcdr_executed=False,initial_mapping_sha256=initial_mapping_sha,
                training_common_users=len(mapping_idx),test_users=len(test_idx),
                fresh_final_checkpoint_sha256=sha(root/'model/final_emcdr.pt'),
                execution=read_json(root/'execution.json'),
                dataset_manifest=f'results/full_reproduction/datasets/task2_beta{beta}.json',
                dataset_sha256=manifest['file_sha256'],
                source_sha256={name:sha(PROJECT/name) for name in ['models.py','run.py','float_embedding.py']},
                selection='Independent minimum MAE and RMSE over epochs; no averaging.')
    write_json(root/'result.json',report)
    write_json(root/'progress.json',dict(stage='completed',beta=beta,updated_at=now()))
    print(json.dumps(metrics),flush=True)


def summarize():
    rows=[]
    paper={20:(1.3524,1.6737),50:(1.4723,1.8000),80:(1.7191,2.1119)}
    for beta in [20,50,80]:
        p=PUBLIC/'trials'/f'beta{beta}_seed2020.json'
        if p.exists():
            d=read_json(p)
            rows.append(dict(beta=beta,**d['metrics'],paper_emcdr_mae=paper[beta][0],paper_emcdr_rmse=paper[beta][1]))
    write_json(PUBLIC/'summary.json',dict(task='Book to Music',method='EMCDR',seed=2020,
                                         completed_runs=len(rows),planned_runs=3,complete=len(rows)==3,
                                         rows=rows,updated_at=now()))
    lines=['<!-- EMCDR_BOOK_MUSIC_FRESH_START -->',
           '| 测试比例 | seed | 新跑 EMCDR MAE | 新跑 EMCDR RMSE | 论文 MAE | 论文 RMSE |',
           '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['beta']}% | 2020 | {r['emcdr_mae']:.4f} | {r['emcdr_rmse']:.4f} | {r['paper_emcdr_mae']:.4f} | {r['paper_emcdr_rmse']:.4f} |")
    lines.append('<!-- EMCDR_BOOK_MUSIC_FRESH_END -->')
    p=PROJECT/'README.md';text=p.read_text(encoding='utf-8-sig');block='\n'.join(lines)
    if '<!-- EMCDR_BOOK_MUSIC_FRESH_START -->' in text:
        text=re.sub(r'<!-- EMCDR_BOOK_MUSIC_FRESH_START -->.*?<!-- EMCDR_BOOK_MUSIC_FRESH_END -->',lambda _:block,text,flags=re.S)
    else:
        text+='\n\n### 图书到音乐 EMCDR 从头重跑结果\n\n'+block+'\n'
    p.write_text(text,encoding='utf-8')
    return len(rows)


def run_queue():
    import msvcrt
    WORK.mkdir(parents=True,exist_ok=True)
    lock=(WORK/'queue.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    for beta in [20,50,80]:
        root=WORK/f'beta{beta}_seed2020';root.mkdir(parents=True,exist_ok=True)
        result_path=PUBLIC/'trials'/f'beta{beta}_seed2020.json'
        if result_path.exists():
            continue
        cfg=config_for(beta);write_json(root/'config.json',cfg)
        public_params=PROJECT/'experiments/emcdr_book_music_fresh_runs'/f'beta{beta}_seed2020.json'
        write_json(public_params,dict(beta=beta,seed=2020,config=cfg,checkpoint_reuse=False,
                                     prototype_reuse=False,method='EMCDR',base_model='MF'))
        commit=publish([str(public_params.relative_to(PROJECT))],f'experiment: 从头训练图书到音乐 EMCDR beta{beta} seed2020')
        write_json(root/'execution.json',dict(commit=commit,started_at=now(),fresh_model=True))
        env=dict(os.environ,PYTHONHASHSEED='2020',PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0')
        home=next(s.split('=',1)[1].strip() for s in TRAIN_PYTHON.parent.parent.joinpath('pyvenv.cfg').read_text().splitlines() if s.startswith('home'))
        env['PATH']=str(Path(home)/'Library/bin')+';'+env['PATH']
        with (root/'train.log').open('ab',buffering=0) as log:
            child=subprocess.Popen([str(TRAIN_PYTHON),'-u',str(Path(__file__).resolve()),'worker','--beta',str(beta)],
                                   cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
            write_json(WORK/'status.json',dict(stage='running',beta=beta,worker_pid=child.pid,
                                              completed_runs=summarize(),planned_runs=3,updated_at=now()))
            code=child.wait()
        if code:
            raise RuntimeError(f'beta{beta} failed with exit code {code}; inspect {root / "train.log"}')
        write_json(result_path,read_json(root/'result.json'))
        completed=summarize()
        publish([str(result_path.relative_to(PROJECT)),'results/emcdr_book_music_fresh/summary.json','README.md'],
                f'results: 图书到音乐 EMCDR beta{beta} seed2020 从头重跑')
        write_json(WORK/'status.json',dict(stage='between_runs',completed_runs=completed,planned_runs=3,updated_at=now()))
    write_json(WORK/'status.json',dict(stage='completed',completed_runs=3,planned_runs=3,updated_at=now()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','worker'])
    parser.add_argument('--beta',type=int,choices=[20,50,80])
    args=parser.parse_args()
    try:
        if args.action=='worker':
            if args.beta is None:
                parser.error('--beta required for worker')
            worker(args.beta)
        else:
            run_queue()
    except Exception:
        WORK.mkdir(parents=True,exist_ok=True)
        write_json(WORK/'failure.json',dict(updated_at=now(),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
