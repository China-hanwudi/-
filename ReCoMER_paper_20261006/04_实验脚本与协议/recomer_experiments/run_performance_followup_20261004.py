"""Finite validation-only search followed by matched-seed confirmation.

Does not alter source architecture, select seeds, load cRBEF packages, or
choose parameters using test scores. Prior test exposure remains disclosed.
"""
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


def save(p, v):
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(v, indent=2, ensure_ascii=False),encoding='utf-8')
    t.replace(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    code = root/'code'
    dest = root/'performance_followup'
    dest.mkdir(exist_ok=False)
    logs = dest/'logs';logs.mkdir()
    lock = (dest/'QUEUE.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    data_root=Path('/root/autodl-tmp/data')
    datasets=[('M3ED_textQwen','cls'),('MELD','cls'),('IEMOCAP','cls'),('MOSEI_full','reg')]
    seeds=[47,59,71,83,97,101,113,127,139,151]
    families=['no_history','history_admission','full_closed_loop','evidence_solo']
    grid={}
    for family in families:
        grid[family]=[dict(lr=lr,bounded_lambda=lam) for lr,lam in
                      itertools.product([5e-5,1e-4,2e-4],
                                        [0.1,0.3] if family.startswith(('full','evidence')) else [None])]
    hashes={str(p.relative_to(code)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in code.rglob('*.py')}
    save(dest/'PLAN.json',dict(scope='CORE_PERFORMANCE_SEARCH_AND_MECHANISM_DIAGNOSTICS_NOT_FULL_RECOMER',
        datasets=datasets,grid=grid,search_seed=43,confirmation_seeds=seeds,
        epochs=20,swa_window=3,selection='valid weighted_f1 for cls; valid mae for reg; deterministic first candidate for ties',
        source_sha256=hashes,search_runs=72,confirmation_runs=160,
        unified_ablations={'seeds':seeds[:3], 'reference':'selected full_closed_loop',
            'changes':{'without_history':['--no-history'],
                       'without_shapley_supervision':['--lambda-u','0'],
                       'uniform_modality_weight':['--bounded-lambda','0']},
            'runs':36,
            'limits':'uniform_modality_weight disables adaptive reweighting, not the bounded parametrization; cRBEF and unbounded-router ablations unavailable'},
        test_exposure='Prior test scores were viewed; new test results are descriptive benchmark evaluations, not blind confirmatory evidence.',
        inference_unit='Matched initialization/training seeds quantify optimization variation only, not independent dataset draws.',
        cRBEF_source_available_in_supplied_folder=False,architecture_changes=False))
    started=time.time()
    def state(phase,**kw):
        row=dict(phase=phase,pid=os.getpid(),seconds=int(time.time()-started),**kw)
        save(dest/'STATUS.json',row);print(json.dumps(row),flush=True)
    def run(command,log):
        with log.open('w',encoding='utf-8') as f:
            subprocess.run(command,cwd=code,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=7200)
    def train(dataset,task,family,candidate,seed,phase,override=None):
        if shutil.disk_usage(root).free<2*1024**3:
            raise RuntimeError('Disk free below 2 GiB; stopped without deleting prior results')
        label=override[0] if override else family
        name=f'{dataset}__{label}__lr{candidate["lr"]}__lam{candidate["bounded_lambda"]}__s{seed}'
        out=dest/phase/name
        if out.exists():raise RuntimeError('Refuse overwrite '+str(out))
        cmd=[sys.executable,'-u','-m','n6.train','--data',str(data_root/dataset/'packed'),
             '--out',str(out),'--task',task,'--seed',str(seed),'--epochs','20',
             '--swa-window','3','--patience','6','--batch-size','256','--device','cuda',
             '--lr',str(candidate['lr']),'--tag','valid_only_performance_search_20261004']
        if family=='no_history':cmd+=['--utility','uniform','--deploy','closed_loop','--no-history']
        elif family=='history_admission':cmd+=['--utility','uniform','--deploy','closed_loop','--history-abstain-variant','utility_softmax']
        else:cmd+=['--utility','shapley','--deploy','closed_loop' if family=='full_closed_loop' else 'solo_weighted',
                    '--gate-architecture','evidence','--gate-detach-inputs','--detach-utility-path',
                    '--bounded-w','--bounded-lambda',str(candidate['bounded_lambda']),
                    '--history-abstain-variant','utility_softmax']
        if override:cmd+=override[1]
        state(phase.upper(),run=name)
        run(cmd,logs/(phase+'__'+name+'.log'))
        result=json.loads((out/'FINAL_RESULT.json').read_text())
        assert result['test_read'] is False and (out/'best.pt').is_file()
        return out,result
    handles=[]
    try:
        state('WAITING_FOR_CORE_QUEUE',training_started=False)
        while True:
            if time.time()-started>86400:raise TimeoutError('Initial queue wait exceeded 24h')
            status=root/'CORE_STATUS.json'
            prior=json.loads(status.read_text()) if status.exists() else {}
            if prior.get('phase')=='FAILED':raise RuntimeError('Prior core queue failed; diagnose before extending it')
            if prior.get('phase')=='COMPLETE':
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
        # Probe GPU health; never change another job or remove existing results.
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True)
        if int(gpu.strip())>=85:raise RuntimeError('GPU temperature too high')
        selected={};rows=[]
        for dataset,task in datasets:
            for family in families:
                best=None
                for candidate in grid[family]:
                    out,r=train(dataset,task,family,candidate,43,'search')
                    metric='weighted_f1' if task=='cls' else 'mae'
                    score=r['best_valid'][metric]
                    rows.append(dict(dataset=dataset,family=family,candidate=candidate,valid_score=score,out=str(out)))
                    save(dest/'SEARCH_ALL_RESULTS.json',rows)
                    if best is None or (score>best['score'] if task=='cls' else score<best['score']):
                        best=dict(candidate=candidate,score=score)
                selected[dataset+'__'+family]=best
        save(dest/'SELECTED_VALID_ONLY.json',selected)
        state('CONFIGURATIONS_FROZEN',test_used_for_selection=False)
        final=[]
        for dataset,task in datasets:
            for family in families:
                for seed in seeds:
                    candidate=selected[dataset+'__'+family]['candidate']
                    out,_=train(dataset,task,family,candidate,seed,'confirmation')
                    final.append((dataset,task,family,seed,out))
        ablations=[]
        for dataset,task in datasets:
            candidate=selected[dataset+'__full_closed_loop']['candidate']
            for label,override in [('without_history',['--no-history']),
                                   ('without_shapley_supervision',['--lambda-u','0']),
                                   ('uniform_modality_weight',['--bounded-lambda','0'])]:
                for seed in seeds[:3]:
                    out,_=train(dataset,task,'full_closed_loop',candidate,seed,'unified_ablations',(label,override))
                    ablations.append((dataset,task,label,seed,out))
        for i,(dataset,task,family,seed,out) in enumerate(final+ablations):
            state('FINAL_TEST',completed_test=i,total_test=len(final+ablations),run=out.name)
            run([sys.executable,'-u','-m','n6.evaluate','--data',str(data_root/dataset/'packed'),
                 '--ckpt',str(out/'best.pt'),'--split','test','--device','cuda',
                 '--out',str(out/'TEST_RESULT.json')],logs/(out.name+'.test.log'))
        # Diagnostics follow frozen selection. Never read them to tune new candidates.
        for dataset,_ in datasets:
            for family in ['full_closed_loop','evidence_solo']:
                ckpt=next(out/'best.pt' for d,t,f,s,out in final if d==dataset and f==family and s==seeds[0])
                state('MECHANISM_DIAGNOSTICS',dataset=dataset,family=family)
                run([sys.executable,'-u',str(root/'collect_mechanism_metrics.py'),
                     '--code',str(code),'--data',str(data_root/dataset/'packed'),
                     '--ckpt',str(ckpt),'--split','test','--device','cuda',
                     '--out',str(dest/'diagnostics'/(dataset+'__'+family))],logs/(dataset+'__'+family+'.diagnostics.log'))
        import numpy as np
        from scipy.stats import t as student_t
        stats=[]
        values={}
        for dataset,task,family,seed,out in final:
            metric='weighted_f1' if task=='cls' else 'mae'
            r=json.loads((out/'TEST_RESULT.json').read_text())
            values.setdefault((dataset,family),{})[seed]=r['metrics'][metric]
        for dataset,task in datasets:
            for family in families:
                x=np.asarray([values[dataset,family][s] for s in seeds])
                row=dict(dataset=dataset,family=family,seed_n=len(seeds),seeds=seeds,mean=float(x.mean()),sd=float(x.std(ddof=1)),values=x.tolist())
                if family not in ['no_history','history_admission']:
                    row['paired_comparisons']={}
                    for baseline in ['no_history','history_admission']:
                        base=np.asarray([values[dataset,baseline][s] for s in seeds])
                        diff=x-base if task=='cls' else base-x
                        half=float(student_t.ppf(.975,len(seeds)-1)*diff.std(ddof=1)/np.sqrt(len(seeds)))
                        row['paired_comparisons'][baseline]=dict(improvement=float(diff.mean()),
                            ci95=[float(diff.mean()-half),float(diff.mean()+half)],
                            positive_seeds=int((diff>0).sum()),definition='paired Student-t CI across training seeds; unadjusted; no significance claim')
                stats.append(row)
        save(dest/'SEED_STATISTICS.json',dict(rows=stats,inference_scope='training-seed variability, conditional on fixed datasets; exploratory, prior test exposure',
                                            multiple_comparisons='intervals unadjusted; no confirmatory significance claims'))
        ablation_rows=[]
        for dataset,task,label,seed,out in ablations:
            metric='weighted_f1' if task=='cls' else 'mae'
            r=json.loads((out/'TEST_RESULT.json').read_text())
            score=r['metrics'][metric]
            ref=values[dataset,'full_closed_loop'][seed]
            ablation_rows.append(dict(dataset=dataset,ablation=label,seed=seed,metric=metric,value=score,
                full_model_value=ref,full_minus_ablated_improvement=ref-score if task=='cls' else score-ref))
        save(dest/'UNIFIED_ABLATION_RESULTS.json',dict(rows=ablation_rows,
            scope='Three matched seeds each; exploratory, reference configuration selected on valid only',
            cRBEF_ablation_available=False,unbounded_router_ablation_available=False))
        state('COMPLETE',search_runs=72,confirmation_runs=160,unified_ablation_runs=36,test_runs=196,
              diagnostic_runs=8,full_recomer_evaluated=False)
    except BaseException as exc:
        state('FAILED',error=repr(exc));raise
    finally:
        for h in handles:h.close()
        lock.close()


if __name__=='__main__':main()
