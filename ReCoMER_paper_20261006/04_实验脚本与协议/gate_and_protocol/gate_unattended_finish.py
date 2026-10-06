"""Server-side completion, bounded retry, paired summaries, and valid-only audit.

This worker needs no desktop connection and never performs test evaluation,
selects new recipes, or edits model/data files.
"""
import ast
import json
import os
import subprocess
import time
from pathlib import Path

EXP = Path('/data/emo/肖田泽科研/模型/model6_innovation_validation_20260930/experiments')
ROOT = EXP/'gate_unattended_20261002'
PY = '/home/emo/anaconda3/envs/torch2.0.1_cuda11.8/bin/python'
DATA = Path('/data/emo/肖田泽科研/数据')
SEEDS = (7,13,17,23,29,37,43,53,71,101)
PILOT = (17,29,43)
VARIANTS = ('constant_task','mlp_task','evidence_task')
JOBS = (
    (1691310, 'launch_gate_confirm_crossdataset.sh', [], 'crossdataset'),
    (1693335, 'launch_gate_feature_screen.sh', ['mosei_full','m3ed_roberta'], 'feature_worker1'),
    (1693336, 'launch_gate_feature_screen.sh', ['meld_roberta','mosi','chsims'], 'feature_worker2'),
)
PACKS = {'mosei':'MOSEI/packed', 'meld':'MELD/packed',
         'm3ed':'M3ED/packed_audio_e2_zh_hubert_large',
         'mosei_full':'MOSEI_full/packed', 'meld_roberta':'MELD_robertaFT/packed',
         'm3ed_roberta':'M3ED_textRobertaFT/packed', 'mosi':'CMU-MOSI/packed',
         'chsims':'CH-SIMS_v2/full_packed'}


def save(name, obj):
    path = ROOT/name
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2),encoding='utf-8')
    temp.replace(path)


def groups():
    yield 'main_mosei', EXP/'gate_router_confirm_mosei_20261002/runs', ('mosei',), SEEDS
    yield 'main_crossdataset', EXP/'gate_router_confirm_crossdataset_20261002/runs', ('meld','m3ed'), SEEDS
    yield 'feature_screen', EXP/'gate_feature_screen_20261002/runs', ('mosei_full','meld_roberta','m3ed_roberta','mosi','chsims'), PILOT


def missing():
    return [str(root/tag/('seed'+str(seed))/v/'FINAL_RESULT.json')
            for _,root,tags,seeds in groups() for tag in tags for seed in seeds for v in VARIANTS
            if not (root/tag/('seed'+str(seed))/v/'FINAL_RESULT.json').exists()]


def running(pid, script):
    path = Path('/proc')/str(pid)/'cmdline'
    try:
        return script.encode() in path.read_bytes()
    except FileNotFoundError:
        return False


def run(cmd, log):
    with (ROOT/log).open('w',encoding='utf-8') as stream:
        subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,check=True)


def formula_ast(path, name):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='UGFModel')
    return ast.dump(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name))


def audit():
    import numpy as np
    import torch
    from n6.data import open_split
    from n6.evaluate import load_checkpoint
    from n6.train_gate import report
    from mosei_gate_benchmark import benchmark_metrics
    torch.set_num_threads(2)
    device = torch.device('cuda')
    old = EXP.parent/'code/n6/model.py'
    new = EXP/'gate_router_20261002/code/n6/model.py'
    for method in ('measure_shapley','deploy_weights'):
        assert formula_ast(old,method)==formula_ast(new,method),method
    rows = []
    for group,root,tags,seeds in groups():
        for tag in tags:
            for seed in seeds:
                run_dir = root/tag/('seed'+str(seed))
                meta = json.loads((run_dir/'evidence_task/RUN_METADATA.json').read_text())
                base,bcfg,_ = load_checkpoint(meta['base_checkpoint'],device)
                ds = open_split(DATA/PACKS[tag],'valid.pt',bcfg.task)
                base_state = base.state_dict()
                y = ds.raw['label'].reshape(-1).to(device)
                for variant in VARIANTS:
                    folder = run_dir/variant
                    saved = json.loads((folder/'FINAL_RESULT.json').read_text())
                    model,cfg,_ = load_checkpoint(folder/'best.pt',device)
                    assert cfg.use_bounded_w and abs(cfg.bounded_lambda-.3)<1e-10
                    assert cfg.utility=='shapley' and cfg.detach_utility_path and cfg.gate_detach_inputs
                    state = model.state_dict()
                    frozen_keys = {k for k in state if not k.startswith('utility_head.')}
                    assert frozen_keys=={k for k in base_state if not k.startswith('utility_head.')}
                    assert all(torch.equal(state[k],base_state[k]) for k in frozen_keys)
                    logits,solos,masks,weights = [],[],[],[]
                    with torch.no_grad():
                        for start in range(0,ds.n,256):
                            b,_ = ds.batch(range(start,min(start+256,ds.n)))
                            b = {k:v.to(device) for k,v in b.items()}
                            out = model(b)
                            logits.append(out['deployed'])
                            solos.append(out['solo_stack'])
                            masks.append(b['modality_mask'])
                            mu = out['utility_mu']
                            w = model.deploy_weights(mu)*b['modality_mask']
                            weights.append(w/w.sum(1,keepdim=True).clamp_min(1e-9))
                    pred,solo,mask,w = map(torch.cat,(logits,solos,masks,weights))
                    assert torch.isfinite(pred).all()
                    key = 'mae' if cfg.task=='reg' else 'weighted_f1'
                    metrics = report(pred,y,cfg)
                    assert abs(metrics[key]-saved['best_valid'][key])<1e-6,(tag,seed,variant)
                    uniform = (mask/mask.sum(1,keepdim=True).clamp_min(1))[:,:,None]*solo
                    uniform = uniform.sum(1)
                    all_present = mask.sum(1)==3
                    assert not all_present.any() or ((w[all_present]>=.7/3-1e-6)&(w[all_present]<=1.3/3+1e-6)).all()
                    if cfg.task=='reg':
                        err = (solo[:,:,0]-y[:,None]).abs().masked_fill(mask<=0,float('inf'))
                        best = err.argmin(1)
                        sample_gain = (uniform[:,0]-y).abs()-(pred[:,0]-y).abs()
                    else:
                        prob = solo.softmax(-1).gather(2,y.long()[:,None,None].expand(-1,3,1)).squeeze(-1)
                        err = (-prob.clamp_min(1e-9).log()).masked_fill(mask<=0,float('inf'))
                        best = err.argmin(1)
                        sample_gain = (pred.argmax(1)==y).float()-(uniform.argmax(1)==y).float()
                    subgroups = {}
                    for j,modality in enumerate(('T','A','V')):
                        keep = (best==j)&(mask.sum(1)>0)
                        subgroups[modality] = dict(n=int(keep.sum()),
                            mean_sample_gain=float(sample_gain[keep].mean()) if keep.any() else None,
                            gate_top_match=float((w[keep].argmax(1)==j).float().mean()) if keep.any() else None)
                    row = dict(group=group,dataset=tag,seed=seed,variant=variant,
                        frozen_base_exact=True,full_forward_matches_saved=True,test_read=False,
                        primary_metric=key,metrics=metrics,
                        mean_weight=w.mean(0).cpu().tolist(),
                        gate_parameters=sum(p.numel() for p in model.utility_head.parameters()),
                        label_informed_best_solo_subgroups=subgroups,
                        subgroup_warning='Posthoc valid diagnostic, not an oracle deployable score or selection rule')
                    if cfg.task=='reg' and tag!='chsims':
                        row['mmsa_metrics']=benchmark_metrics(y.cpu().numpy(),pred[:,0].cpu().numpy())
                    rows.append(row)
                    del model,state,pred,solo,mask,w
                del base,base_state
                print('AUDITED',group,tag,seed,flush=True)
                save('checkpoint_audit.json',dict(status='RUNNING',rows=rows,test_read=False))
    save('checkpoint_audit.json',dict(status='AUDIT_COMPLETE',n=len(rows),rows=rows,
        original_shapley_and_centered_bounded_formula_ast_equal=True,test_read=False))


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    save('STATUS.json',dict(status='WAITING_FOR_REMOTE_QUEUES',jobs=JOBS,test_read=False))
    deadline = time.time()+24*3600
    while any(running(pid,script) for pid,script,_,_ in JOBS):
        if time.time()>deadline:
            raise RuntimeError('24-hour queue deadline exceeded; inspect logs, no jobs killed')
        time.sleep(20)
    # One bounded retry of idempotent queues; completed variants are skipped.
    if missing():
        save('STATUS.json',dict(status='BOUNDED_RETRY',missing=missing(),test_read=False))
        for _,script,args,name in JOBS:
            run(['bash',str(ROOT/script)]+args,'retry_'+name+'.log')
    if missing():
        raise RuntimeError('Runs still incomplete after one retry: '+str(missing()))
    save('STATUS.json',dict(status='SUMMARIZING_AND_AUDITING',test_read=False))
    for group,root,tags,seeds in groups():
        cmd = [PY,str(ROOT/'gate_multidataset_summary.py'),'--root',str(root),
               '--datasets',','.join(tags),'--out',str(ROOT/(group+'_summary.json'))]
        if seeds==PILOT:
            cmd.append('--screen')
        run(cmd,group+'_summary.log')
    audit()
    save('STATUS.json',dict(status='ALL_FINISHED',main_seeds=10,feature_screen_seeds=3,
        main_datasets=['MOSEI','MELD','M3ED'],additional_datasets=['CMU-MOSI','CH-SIMS_v2'],
        strong_feature_packs=['MOSEI_full','MELD_robertaFT','M3ED_textRobertaFT'],
        gate_fits_total=135,test_read=False,sota_claim=False,
        limitation='valid-only results; additional and strong-feature runs are exploratory, no official-test SOTA verdict',
        completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    print('ALL_FINISHED',flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        save('STATUS.json',dict(status='FAILED_REQUIRES_REVIEW',error=repr(exc),test_read=False))
        raise
