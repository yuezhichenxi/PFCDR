"""Approved EMCDR LR grid: 12 new fresh runs plus 3 existing fresh controls."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_emcdr_book_music_fresh as fresh
from run_book_music_k_sweep import git,now,read_json,sha,write_json

PROJECT=Path(__file__).resolve().parents[1]
WORK=PROJECT.parent/'ccfa-workfiles/experiments/pfcdr-emcdr-book-music-lr-grid-20261117'
PUBLIC=PROJECT/'results/emcdr_book_music_lr_grid'
PLAN=PROJECT/'experiments/emcdr_book_music_lr_grid_plan.json'
TRAIN_PYTHON=WORK.parent/'pfcdr-book-movie-20/train-env/Scripts/python.exe'
LRS=[0.001,0.005,0.01,0.02,0.1]
BETAS=[20,50,80]
PAPER={20:(1.3524,1.6737),50:(1.4723,1.8000),80:(1.7191,2.1119)}


def job_id(beta,lr):
    if beta not in BETAS or lr not in LRS:
        raise ValueError('Configuration is outside the approved grid')
    return f'lr{str(lr).replace(".","p")}_beta{beta}_seed2020'


def config_for(beta,lr):
    cfg=fresh.config_for(beta)
    cfg['lr']=lr
    cfg['model_root']=(WORK/job_id(beta,lr)/'model').as_posix()+'/'
    return cfg


def publish(paths,title):
    if git('diff','--cached','--name-only'):
        raise RuntimeError('Refusing to mix unrelated staged changes')
    git('add','--',*paths)
    if git('diff','--cached','--name-only'):
        message=WORK/'commit-message.txt'
        message.write_text(title+'\n\nMF Book-to-Music EMCDR LR grid; seed=2020, '
                           '10 epochs per stage, original batches 512/512/128, eval=256. '
                           'Full configuration and source of existing controls are recorded.\n',encoding='utf-8')
        git('commit','--file',str(message))
    commit=git('rev-parse','HEAD')
    try:
        detail=git('-c','credential.interactive=false','push','origin','main')
        write_json(WORK/'push-status.json',dict(ok=True,commit=commit,detail=detail,updated_at=now()))
    except Exception as error:
        write_json(WORK/'push-status.json',dict(ok=False,commit=commit,detail=str(error),updated_at=now()))
    return commit


def summarize():
    rows=[]
    for beta in BETAS:
        for lr in LRS:
            path=PUBLIC/'trials'/f'{job_id(beta,lr)}.json'
            if not path.exists():
                continue
            d=read_json(path)
            row=dict(beta=beta,lr=lr,success=d.get('complete',False),
                     run_kind=d.get('run_kind','new_fresh_training'),
                     paper_mae=PAPER[beta][0],paper_rmse=PAPER[beta][1])
            if row['success']:
                row.update(mae=d['metrics']['emcdr_mae'],rmse=d['metrics']['emcdr_rmse'])
                row.update(delta_mae=row['mae']-row['paper_mae'],delta_rmse=row['rmse']-row['paper_rmse'])
            else:
                row['error']=d.get('error','Run failed; inspect trial report')
            rows.append(row)
    best={}
    for beta in BETAS:
        candidates=[r for r in rows if r['beta']==beta and r['success']]
        if candidates:
            best[str(beta)]={metric:{'lr':min(candidates,key=lambda r:r[metric])['lr'],
                                     'value':min(r[metric] for r in candidates)} for metric in ['mae','rmse']}
    successful=sum(r['success'] for r in rows)
    write_json(PUBLIC/'summary.json',dict(task='Book to Music',method='EMCDR',base_model='MF',seed=2020,
                                         finished_configurations=len(rows),successful_configurations=successful,
                                         planned_configurations=15,adopted_controls=3,new_training_runs=12,
                                         complete=len(rows)==15,all_successful=successful==15,
                                         rows=rows,best_by_ratio=best,updated_at=now()))
    lines=['<!-- EMCDR_BOOK_MUSIC_LR_GRID_START -->',
           '| 测试比例 | 学习率 | MAE | RMSE | 与论文 MAE 差值 | 与论文 RMSE 差值 | 来源 |',
           '| ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for r in rows:
        origin='已完成的从头训练对照' if r['run_kind']=='adopted_fresh_baseline' else '本轮从头训练'
        if r['success']:
            lines.append(f"| {r['beta']}% | {r['lr']} | {r['mae']:.4f} | {r['rmse']:.4f} | {r['delta_mae']:+.4f} | {r['delta_rmse']:+.4f} | {origin} |")
        else:
            lines.append(f"| {r['beta']}% | {r['lr']} | 失败 | 失败 | 不适用 | 不适用 | {origin} |")
    lines.append('<!-- EMCDR_BOOK_MUSIC_LR_GRID_END -->')
    readme=PROJECT/'README.md';text=readme.read_text(encoding='utf-8-sig');block='\n'.join(lines)
    if '<!-- EMCDR_BOOK_MUSIC_LR_GRID_START -->' in text:
        text=re.sub(r'<!-- EMCDR_BOOK_MUSIC_LR_GRID_START -->.*?<!-- EMCDR_BOOK_MUSIC_LR_GRID_END -->',lambda _:block,text,flags=re.S)
    else:
        text+='\n\n### 图书到音乐 EMCDR 学习率扫描实际结果\n\n'+block+'\n'
    readme.write_text(text,encoding='utf-8')
    return len(rows)


def adopt_controls():
    for beta in BETAS:
        source=PROJECT/f'results/emcdr_book_music_fresh/trials/beta{beta}_seed2020.json'
        dest=PUBLIC/'trials'/f'{job_id(beta,0.01)}.json'
        if dest.exists():
            continue
        d=read_json(source)
        assert d['complete'] and d['checkpoint_loaded'] is False
        assert d['source_and_target_trained_from_scratch'] is True
        assert d['config']['lr']==0.01 and d['config']['epoch']==10 and d['seed']==2020
        p=d['config']['src_tgt_pairs']['2']
        assert [p[k] for k in ['batchsize_src','batchsize_tgt','batchsize_map','batchsize_test']]==[512,512,128,256]
        d.update(learning_rate=0.01,run_kind='adopted_fresh_baseline',new_training_in_this_grid=False,
                 adopted_from=str(source.relative_to(PROJECT)),adopted_report_sha256=sha(source))
        write_json(dest,d)
    summarize()
    publish(['results/emcdr_book_music_lr_grid','README.md'],
            'experiment: 纳入三组已完成的lr0.01从头训练对照')


def worker(beta,lr):
    root=WORK/job_id(beta,lr)
    fresh.worker(beta,job_root=root)
    d=read_json(root/'result.json')
    assert d['config']['lr']==lr and not d['checkpoint_loaded']
    d.update(learning_rate=lr,run_kind='new_fresh_training',new_training_in_this_grid=True)
    write_json(root/'result.json',d)


def run_queue():
    import msvcrt
    WORK.mkdir(parents=True,exist_ok=True)
    lock=(WORK/'queue.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    adopt_controls()
    jobs=[(beta,lr) for lr in LRS if lr!=0.01 for beta in BETAS]
    for number,(beta,lr) in enumerate(jobs,1):
        name=job_id(beta,lr);root=WORK/name;root.mkdir(parents=True,exist_ok=True)
        public_result=PUBLIC/'trials'/f'{name}.json'
        if public_result.exists():
            continue
        cfg=config_for(beta,lr);write_json(root/'config.json',cfg)
        params=PROJECT/'experiments/emcdr_book_music_lr_grid_runs'/f'{name}.json'
        write_json(params,dict(task='2',method='EMCDR',base_model='MF',beta=beta,seed=2020,
                               learning_rate=lr,optimizer_learning_rates={'source':lr,'target':lr,'mapping':lr},
                               config=cfg,checkpoint_reuse=False,run_kind='new_fresh_training'))
        commit=publish([str(params.relative_to(PROJECT))],f'experiment: EMCDR图书到音乐 lr{lr} beta{beta} seed2020')
        write_json(root/'execution.json',dict(commit=commit,started_at=now(),fresh_model=True,learning_rate=lr))
        env=dict(os.environ,PYTHONHASHSEED='2020',PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0')
        home=next(s.split('=',1)[1].strip() for s in TRAIN_PYTHON.parent.parent.joinpath('pyvenv.cfg').read_text().splitlines() if s.startswith('home'))
        env['PATH']=str(Path(home)/'Library/bin')+';'+env['PATH']
        with (root/'train.log').open('ab',buffering=0) as log:
            child=subprocess.Popen([str(TRAIN_PYTHON),'-u',str(Path(__file__).resolve()),'worker','--beta',str(beta),'--lr',str(lr)],
                                   cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
            write_json(WORK/'status.json',dict(stage='running',job=name,beta=beta,learning_rate=lr,
                                              new_run_number=number,planned_new_runs=12,worker_pid=child.pid,
                                              finished_configurations=summarize(),planned_configurations=15,updated_at=now()))
            code=child.wait()
        if code==0:
            write_json(public_result,read_json(root/'result.json'))
        else:
            failed=dict(complete=False,beta=beta,seed=2020,learning_rate=lr,run_kind='new_fresh_training',
                        error=f'Worker exit {code}; see execution logs',completed_at=now())
            if (root/'partial-results.json').exists():
                failed['partial_results']=read_json(root/'partial-results.json')
            if (root/'failure.json').exists():
                failed['failure']=read_json(root/'failure.json')
            write_json(public_result,failed)
        finished=summarize()
        publish([str(public_result.relative_to(PROJECT)),'results/emcdr_book_music_lr_grid/summary.json','README.md'],
                f'results: EMCDR图书到音乐 lr{lr} beta{beta} seed2020')
        write_json(WORK/'status.json',dict(stage='between_runs',finished_configurations=finished,planned_configurations=15,
                                          completed_new_runs=number,planned_new_runs=12,updated_at=now()))
    s=read_json(PUBLIC/'summary.json')
    write_json(WORK/'status.json',dict(stage='completed',finished_configurations=s['finished_configurations'],
                                      successful_configurations=s['successful_configurations'],planned_configurations=15,
                                      completed_new_runs=12,adopted_controls=3,updated_at=now()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','worker'])
    parser.add_argument('--beta',type=int,choices=BETAS)
    parser.add_argument('--lr',type=float,choices=LRS)
    args=parser.parse_args()
    try:
        if args.action=='worker':
            if args.beta is None or args.lr is None:
                parser.error('worker requires --beta and --lr')
            worker(args.beta,args.lr)
        else:
            run_queue()
    except Exception:
        WORK.mkdir(parents=True,exist_ok=True)
        path=WORK/job_id(args.beta,args.lr)/'failure.json' if args.action=='worker' else WORK/'failure.json'
        write_json(path,dict(updated_at=now(),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
