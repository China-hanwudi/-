"""Targeted fixed-architecture data/regularization search for M3ED and MELD."""
import argparse
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def save(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2),encoding='utf-8');tmp.replace(p)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output-root',type=Path);args=ap.parse_args()
    root=args.root.resolve();dest=args.output_root.resolve() if args.output_root else root/'data_followup'
    dest.mkdir(parents=True,exist_ok=False)
    code=root/'code';logs=dest/'logs';logs.mkdir()
    wrapper=root/'data_pipeline_probe.py'
    own=(dest/'QUEUE.lock').open('a+');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    modes=['identity','train_zscore','train_zclip5','train_center_l2']
    recipes={'original':[], 'regularized_cosine':['--dropout','0.25','--weight-decay','0.03',
                 '--warmup-epochs','2','--cosine-epochs','20','--min-lr','1e-6',
                 '--epochs','30','--patience','8','--loss-patience','0']}
    families=['full_closed_loop','history_admission']
    seeds=[47,59,71,83,97,101,113,127,139,151]
    datasets=['M3ED_textQwen','MELD']
    hashes={str(p.relative_to(code)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code.rglob('*.py')}
    save(dest/'PLAN.json',dict(datasets=datasets,modes=modes,recipes=recipes,families=families,
        screening_seed=43,screening_runs=32,confirmation_seeds=seeds,confirmation_runs=40,
        selection='valid only; same search grid for full model and history-admission baseline',
        original_pack_modified=False,labels_changed=False,samples_removed=0,split_changed=False,
        model_architecture_changed=False,dimension_or_layers_changed=False,
        normalizer_fit_split='train only',prior_test_exposure=True,
        source_sha256=hashes,wrapper_sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest(),
        inherited_parameters='Selected valid-only lr and bounded_lambda from earlier performance search',
        missing_experiments=['cRBEF and complete ReCoMER four-way comparisons require missing expert assets'],
        total_time_limit_seconds=172800,minimum_disk_free_GiB=2))
    began=time.time();handles=[]
    def state(phase,**kw):
        r=dict(phase=phase,pid=os.getpid(),seconds=int(time.time()-began),**kw)
        save(dest/'STATUS.json',r);print(json.dumps(r),flush=True)
    def run(cmd,log):
        with log.open('w',encoding='utf-8') as f:
            subprocess.run(cmd,cwd=code,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=7200)
    def room():
        if shutil.disk_usage(dest).free<2*1024**3:
            raise RuntimeError('Disk free below 2GiB; expansion or verified archival required; no user data deleted')
    def train(dataset,family,candidate,seed,phase,base):
        room();data=Path('/root/autodl-tmp/data')/dataset/'packed';moments=dest/(dataset+'.moments.pt')
        name=f'{dataset}__{family}__{candidate["mode"]}__{candidate["recipe"]}__s{seed}'
        out=dest/phase/name
        command=[sys.executable,'-u',str(wrapper),'--code',str(code),'--data',str(data),
                 '--action','train','--mode',candidate['mode'],'--moments',str(moments),'--out',str(out),'--',
                 '--task','cls','--seed',str(seed),'--epochs','20','--swa-window','3','--patience','6',
                 '--batch-size','256','--device','cuda','--lr',str(base['lr']),
                 '--history-abstain-variant','utility_softmax','--deploy','closed_loop',
                 '--tag','fixed_architecture_trainfit_data_search_20261004']
        if family=='full_closed_loop':command+=['--utility','shapley','--gate-architecture','evidence',
                    '--gate-detach-inputs','--detach-utility-path','--bounded-w','--bounded-lambda',str(base['bounded_lambda'])]
        else:command+=['--utility','uniform']
        command+=recipes[candidate['recipe']]
        state(phase.upper(),run=name)
        run(command,logs/(phase+'__'+name+'.log'))
        r=json.loads((out/'FINAL_RESULT.json').read_text());assert r['test_read'] is False
        return out,r
    try:
        # Numerical/data audit is CPU-only and may run while the prior GPU queue trains.
        for dataset in datasets:
            data=Path('/root/autodl-tmp/data')/dataset/'packed'
            run([sys.executable,str(wrapper),'--code',str(code),'--data',str(data),
                 '--action','audit','--out',str(dest/(dataset+'.audit.json'))],logs/(dataset+'.audit.log'))
            run([sys.executable,str(wrapper),'--code',str(code),'--data',str(data),
                 '--action','fit','--moments',str(dest/(dataset+'.moments.pt'))],logs/(dataset+'.fit.log'))
        state('WAITING_FOR_PERFORMANCE_QUEUE',training_started=False)
        while True:
            if time.time()-began>172800:raise TimeoutError('Queue waiting exceeded 48h')
            status=root/'performance_followup'/'STATUS.json'
            p=json.loads(status.read_text()) if status.exists() else {}
            if p.get('phase')=='FAILED':raise RuntimeError('Prior search failed; investigate rather than ignore it')
            if p.get('phase')=='COMPLETE':
                try:
                    for name in ['GPU_RESEARCH.lock','FUSION_RESEARCH.lock']:
                        h=(Path('/root/autodl-tmp/fusion_research_automation_20261003')/name).open('a+')
                        handles.append(h);fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
                    if active:raise BlockingIOError('GPU occupied')
                    break
                except BlockingIOError:
                    for h in handles:h.close()
                    handles.clear()
            time.sleep(30)
        assert hashes=={str(p.relative_to(code)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code.rglob('*.py')}
        prior=json.loads((root/'performance_followup'/'SELECTED_VALID_ONLY.json').read_text())
        selected={};all_rows=[]
        for dataset in datasets:
            for family in families:
                base=prior[dataset+'__'+family]['candidate'];best=None
                for mode,recipe in itertools.product(modes,recipes):
                    candidate=dict(mode=mode,recipe=recipe)
                    out,r=train(dataset,family,candidate,43,'screen',base)
                    score=r['best_valid']['weighted_f1']
                    all_rows.append(dict(dataset=dataset,family=family,candidate=candidate,score=score,out=str(out)))
                    save(dest/'ALL_VALID_RESULTS.json',all_rows)
                    if best is None or score>best['score']:best=dict(candidate=candidate,score=score,base=base)
                selected[dataset+'__'+family]=best
        save(dest/'SELECTED_VALID_ONLY.json',selected)
        state('DATA_RECIPES_FROZEN',test_used_for_selection=False)
        final=[]
        for dataset in datasets:
            for family in families:
                choice=selected[dataset+'__'+family]
                for seed in seeds:
                    out,_=train(dataset,family,choice['candidate'],seed,'confirmation',choice['base'])
                    final.append((dataset,family,seed,out,choice))
        rows=[]
        for i,(dataset,family,seed,out,choice) in enumerate(final):
            room();state('FINAL_TEST',run=out.name,completed_test=i)
            run([sys.executable,str(wrapper),'--code',str(code),
                 '--data',str(Path('/root/autodl-tmp/data')/dataset/'packed'),'--action','evaluate',
                 '--mode',choice['candidate']['mode'],'--moments',str(dest/(dataset+'.moments.pt')),
                 '--out',str(out/'TEST_RESULT.json'),'--','--ckpt',str(out/'best.pt'),
                 '--split','test','--device','cuda'],logs/(out.name+'.test.log'))
            report=json.loads((out/'TEST_RESULT.json').read_text())
            rows.append(dict(dataset=dataset,family=family,seed=seed,candidate=choice,
                             metrics=report['metrics'],test_exposure='previously viewed benchmark; descriptive'))
            save(dest/'ALL_TEST_RESULTS.json',rows)
        for dataset,family,seed,out,choice in final:
            if family=='full_closed_loop' and seed==seeds[0]:
                state('MECHANISM_DIAGNOSTICS',dataset=dataset,family=family)
                run([sys.executable,str(root/'collect_mechanism_metrics.py'),
                     '--code',str(code),'--data',str(Path('/root/autodl-tmp/data')/dataset/'packed'),
                     '--ckpt',str(out/'best.pt'),'--split','test','--device','cuda',
                     '--out',str(dest/'diagnostics'/dataset)],logs/(dataset+'.diagnostics.log'))
        state('COMPLETE',screening_runs=32,confirmation_runs=40,test_runs=40,mechanism_runs=2,
              architecture_changed=False,original_pack_changed=False)
    except BaseException as exc:
        state('FAILED',error=repr(exc));raise
    finally:
        for h in handles:h.close()
        own.close()


if __name__=='__main__':main()
