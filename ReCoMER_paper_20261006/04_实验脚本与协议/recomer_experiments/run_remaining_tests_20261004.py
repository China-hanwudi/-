"""Frozen evaluation and matched-fold ablations of the actual completed ReCoMER.

Run in an isolated remote directory. Production code and previous workflows
are read-only. Test results never select configurations, checkpoints or seeds.
The five retrained controls use a documented runtime override where the
production configuration intentionally disallows incomplete architectures.
"""
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

SEEDS = [43, 47, 59]
ARMS = {
    "no_shapley_supervision": "Set lambda_u=0; retain architecture and measured teacher for matched compute.",
    "unbounded_softmax_weights": "Replace bounded mapping with the existing softmax/floor mapping; same evidence router.",
    "no_history_abstention": "Set history_abstain_variant=none; retain history pooling and feedback.",
    "no_contribution_token_feedback": "Omit token-weight scaling in joint forward and Shapley measurement; retain history feedback.",
    "no_history_residual_feedback": "HistoryAttnPool.apply returns current embedding unchanged; retain contribution token scaling.",
}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix('.tmp')
    q.write_text(json.dumps(v, indent=2, ensure_ascii=False), encoding='utf-8')
    q.replace(p)


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def install_arm(arm):
    """Process-local override, never change production source or saved cfg."""
    from n6.model import UGFModel, HistoryAttnPool
    if arm == 'unbounded_softmax_weights':
        original = UGFModel.deploy_weights
        def unbounded(self, mu, sigma=None):
            old = self.cfg.use_bounded_w
            try:
                self.cfg.use_bounded_w = False
                return original(self, mu, sigma)
            finally:
                self.cfg.use_bounded_w = old
        UGFModel.deploy_weights = unbounded
    elif arm == 'no_contribution_token_feedback':
        original = UGFModel.build_tokens
        def unweighted(self, *a, **kw):
            kw['token_weight'] = None
            return original(self, *a, **kw)
        UGFModel.build_tokens = unweighted
        original_subset = UGFModel.joint_on_subset
        def subset(self, *a, **kw):
            kw['token_weight'] = None
            return original_subset(self, *a, **kw)
        UGFModel.joint_on_subset = subset
    elif arm == 'no_history_residual_feedback':
        HistoryAttnPool.apply = lambda self, cur, pooled, gate: cur
    elif arm not in ('full', 'no_shapley_supervision', 'no_history_abstention'):
        raise ValueError('unknown arm: ' + arm)


def train_worker(arm):
    install_arm(arm)
    from n6.train import main
    return main(sys.argv[4:])


def main():
    import numpy as np
    import torch
    root = Path(sys.argv[1]).resolve()
    previous = Path('/root/recomer_complete_20261004_side')
    grid = Path('/root/recomer_validation_20261004_side')
    assets = Path('/root/autodl-tmp/fusion_research_automation_20261003')
    source = assets / 'round017_repair1'
    other_test = assets / 'side_official_test_20261004'
    code = root / 'code'
    sys.path.insert(0, str(code))
    from n6.data import open_split, open_final_test
    from n6.evaluate import load_checkpoint
    from n6.metrics import classification_report, regression_report
    from n6.recomer import ReCoMER, mhnou_reference_outputs, combine_inputs
    from n6.crbef_expert import CRBEFExpert, CRBEFFeatureStore, M3ED_CLASSES
    from n6.crbef_fusion import LocalCurrentConfig, LocalCurrentFusion, fit_local_current
    from tools.train_recomer_fusion import load_inputs, four_way
    torch.set_num_threads(2)
    started = time.monotonic()
    handles, attempts = [], []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')

    def status(phase, **kw):
        v = dict(phase=phase, pid=os.getpid(), seconds=round(time.monotonic()-started),
                 attempts=len(attempts), failures=sum(a['returncode'] != 0 for a in attempts),
                 previous_paused_workflow_resumed=False, **kw)
        save(root / 'STATUS.json', v)
        print(json.dumps(v), flush=True)

    def run(cmd, name, required=True):
        if shutil.disk_usage(root).free < 2 << 30:
            raise RuntimeError('disk reserve below 2 GiB')
        outlog = root / 'logs' / (name+'.log')
        outlog.parent.mkdir(exist_ok=True)
        t = time.monotonic()
        with outlog.open('w') as f:
            cp = subprocess.run(cmd, cwd=code, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=3600)
        attempts.append(dict(name=name, command=cmd, returncode=cp.returncode,
                             seconds=time.monotonic()-t, log=str(outlog)))
        save(root / 'ATTEMPTS.json', attempts)
        if required and cp.returncode:
            raise RuntimeError(f'{name} failed; inspect {outlog}')
        return cp.returncode

    def metrics(p, y):
        return classification_report(np.log(np.maximum(p, 1e-9)), y, 7, M3ED_CLASSES)

    def export(model, ds, ids, cr_cache):
        lut = {str(v): i for i, v in enumerate(ds.raw['ids'])}
        ci = CRBEFFeatureStore._lookup(cr_cache['ids'], 'expert cache')
        assert len(set(ids)) == len(ids) and set(ids).issubset(lut) and set(ids).issubset(ci)
        bins = {k: [] for k in ('P', 'C', 'L0', 'Lh', 'M')}
        for start in range(0, len(ids), 128):
            cur = ids[start:start+128]
            b, _ = ds.batch([lut[v] for v in cur])
            b = {k: v.cuda() for k, v in b.items() if k != 'history_label'}
            mh = mhnou_reference_outputs(model, b)
            cp = torch.from_numpy(cr_cache['P'][[ci[v] for v in cur]]).cuda()
            p, c = combine_inputs(mh, dict(CRBEF=cp[:, 0], TAV=cp[:, 2]))
            out = dict(P=p, **mh)
            for k in bins:
                bins[k].append(out[k].cpu().numpy())
        out = {k: np.concatenate(v) for k, v in bins.items()}
        out.update(ids=np.asarray(ids), source=np.asarray([v.split('_')[1] for v in ids]),
                   class_names=np.asarray(M3ED_CLASSES), y_true=ds.raw['label'][[lut[v] for v in ids]].numpy())
        return out

    def fit_fuser(fit, cal, out, seed, no_c=False):
        fit, cal = copy.deepcopy(fit), copy.deepcopy(cal)
        assert set(fit['ids']).isdisjoint(cal['ids']) and set(fit['source']).isdisjoint(cal['source'])
        if no_c:
            fit['C'] *= 0
            cal['C'] *= 0
        torch.manual_seed(seed)
        fuser, fit_stats = fit_local_current(torch.from_numpy(fit['P']), torch.from_numpy(fit['C']),
                torch.from_numpy(fit['y_true']), LocalCurrentConfig(7), epochs=40)
        best, chosen, choices = None, 0.0, []
        for eta in (0.0, 0.25, 0.5, 1.0):
            fuser.cfg.eta = eta
            report, _ = four_way(fuser, cal)
            r = report['ReCoMER']
            key = (r['weighted_f1'], r['macro_f1'], r['accuracy'])
            choices.append(dict(eta=eta, metrics=r))
            if best is None or key > best:
                best, chosen = key, eta
        fuser.cfg.eta = chosen
        out.mkdir(parents=True, exist_ok=False)
        torch.save(dict(cfg=fuser.cfg.__dict__, model=fuser.state_dict(), seed=seed,
                        class_names=list(M3ED_CLASSES)), out/'local_current.pt')
        save(out/'TRAIN_RESULT.json', dict(seed=seed, eta=chosen, choices=choices,
             fit_stats=fit_stats, fit_rows=len(fit['ids']), cal_rows=len(cal['ids']), no_C=no_c,
             selection='CAL only, no test-dependent selection', full_system_OOF=False))
        return fuser

    try:
        for name in ('GPU_RESEARCH.lock', 'FUSION_RESEARCH.lock'):
            h = (assets/name).open('a+')
            fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
            handles.append(h)
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip():
            raise RuntimeError('GPU occupied; do not interrupt existing work')
        assert not (root/'PLAN.json').exists(), 'one-shot queue; preserve previous runs'
        # Select both baseline learning rates exclusively by saved valid results.
        selection = []
        for dataset, task in [('M3ED_textQwen','cls'), ('MELD','cls'), ('IEMOCAP','cls'), ('MOSEI_full','reg')]:
            for arm in ('mhnou_final','no_history_reference'):
                choices = []
                for lr in (1e-4, 2e-4):
                    runs = [grid/'runs'/f'{dataset}__{arm}__s{s}__lr{lr:g}' for s in SEEDS]
                    vals = [load(p/'FINAL_RESULT.json')['best_valid'] for p in runs]
                    key = 'weighted_f1' if task == 'cls' else 'mae'
                    choices.append(dict(lr=lr, mean=float(np.mean([v[key] for v in vals])), runs=[str(p) for p in runs]))
                selected = sorted(choices, key=lambda c: (-c['mean'] if task=='cls' else c['mean'], c['lr']))[0]
                selection.append(dict(dataset=dataset, task=task, arm=arm, selected=selected, choices=choices))
        initial_assets = [previous/'recomer'/str(s)/'recomer.pt' for s in SEEDS]
        initial_assets += [Path(p)/'best.pt' for q in selection for p in q['selected']['runs']]
        orig = assets/'round032_same_cr_valid'
        base = Path('/root/autodl-tmp/models/Qwen3-8B-AWQ')
        adapter = Path('/root/autodl-tmp/crbef_full_package/weights/text_adapter_m3ed_s43')
        old = Path('/root/autodl-tmp/classmate_temporaln3_newbench_20260913/m3ed/packed')
        rich_test = Path('/root/autodl-tmp/multimodal_newbench_20260912/runs/m3ed_e2v_plus_audio_test/test_features.pt')
        initial_assets += [adapter/'adapter_model.safetensors', adapter/'adapter_config.json',
                          other_test/'inputs/test_tokens.npz', other_test/'inputs/AUDIT.json',
                          orig/'full_v2/features.npz', old/'test.pt', rich_test,
                          source/'CONTRACTS.json']
        for s in SEEDS:
            initial_assets += [previous/'inputs'/str(s)/n for n in ('FIT.npz','final_fusion_cal.npz')]
        frozen = dict(created=time.time(), seeds=SEEDS, production_model='original CR17 + latest MHnoU + local_current',
             cRBEF_seed_fixed=17, main_default_seed=43, ablations=ARMS, mh_lr=2e-4, mh_epochs=20,
             mh_folds=['inner0','inner1','inner2','inner3','final'], ablation_mh_fits=75,
             fusion_epochs=40, eta_candidates=[0.0,0.25,0.5,1.0], bootstrap_resamples=1000,
             baseline_selection=selection, asset_sha256={str(p):sha(p) for p in initial_assets},
             code_sha256={str(p.relative_to(code)):sha(p) for p in code.rglob('*.py')},
             harness_sha256=sha(Path(__file__)), teacher_base_sha256={str(p):sha(p) for p in base.glob('*.safetensors')},
             full_system_OOF=False, prior_benchmark_test_exposure=True,
             no_test_selection=True, masks=['T','A','V','TA','TV','AV'],
             robustness_scope='feature-level zero/mask stress, no encoder retraining; not a missing-raw-modality benchmark',
             limitations=['fixed CR17 teacher exposure remains; not fully OOF',
                          'CR weights available only for M3ED; other datasets evaluate MHnoU only',
                          'three seeds vary MHnoU and fuser; original cRBEF is fixed',
                          'runtime ablations require this harness to reproduce',
                          'no contribution feedback and no history feedback are separate FCCR subcomponent controls',
                          'external literature baselines are not reproduced by this queue'])
        save(root/'PLAN.json', frozen)
        run([sys.executable,'-m','unittest','discover','-s','tests','-v'], 'unit_tests')
        status('RECOVERING_ORIGINAL_CR17_TEST_FEATURES')
        # Reuse only identity-qualified prompt tokens, NEVER the other task's CR weights/h.
        assert load(other_test/'inputs/AUDIT.json')['status']=='PASS_COMPONENT_IDENTITY_AND_PROMPT_QUALIFICATION'
        old_frozen = load(orig/'FROZEN_v2.json')
        for p in (adapter/'adapter_model.safetensors', adapter/'adapter_config.json'):
            assert sha(p)==old_frozen['inputs'][str(p)]
        from transformers import AutoModelForCausalLM, AwqConfig
        from peft import PeftModel
        with np.load(other_test/'inputs/test_tokens.npz', allow_pickle=False) as z:
            tokens = {k:z[k].copy() for k in ('ids','input_ids','offsets')}
        with np.load('/root/research_rgfv4_m3ed_001/text_prep_001/dev_tokens.npz', allow_pickle=False) as z:
            qual = {k:z[k].copy() for k in ('ids','input_ids','offsets')}
        lm = AutoModelForCausalLM.from_pretrained(str(base), dtype=torch.float16, device_map='cuda:0',
             quantization_config=AwqConfig(bits=4, group_size=128, zero_point=True, backend='auto_trainable'), local_files_only=True)
        lm = PeftModel.from_pretrained(lm, str(adapter)).eval()
        lm.config.use_cache = False
        for p in lm.parameters(): p.requires_grad_(False)
        def hidden(bank, n):
            values=[]
            with torch.no_grad():
                for st in range(0,n,16):
                    ix=range(st,min(st+16,n))
                    seq=[bank['input_ids'][bank['offsets'][i]:bank['offsets'][i+1]] for i in ix]
                    lens=np.array([len(v) for v in seq])
                    inp=np.full((len(seq),int(lens.max())),151643,np.int64);att=np.zeros_like(inp)
                    for i,v in enumerate(seq):inp[i,:len(v)]=v;att[i,:len(v)]=1
                    res=lm(input_ids=torch.from_numpy(inp).cuda(),attention_mask=torch.from_numpy(att).cuda(),output_hidden_states=True)
                    values.append(res.hidden_states[-1][torch.arange(len(seq),device='cuda'),torch.from_numpy(lens-1).cuda()].float().cpu().numpy())
                    del res
            return np.concatenate(values)
        qh = hidden(qual,32)
        with np.load(orig/'full_v2/features.npz',allow_pickle=False) as z:
            reference={k:z[k][:32].copy() for k in ('ids','h','a','v','pa','pv')}
        assert np.array_equal(reference['ids'],qual['ids'][:32])
        herror=float(np.max(np.abs(qh-reference['h'])))
        # Saved valid recovery is already qualified against all 2821 archived head probabilities.
        assert herror<=1e-5, ('original valid hidden replay changed',herror)
        test_h = hidden(tokens,len(tokens['ids']))
        del lm
        torch.cuda.empty_cache()
        reader_path=assets/'round036_downstream_preparation/cr_contract/lazy_pt.py'
        spec=importlib.util.spec_from_file_location('remaining_lazy_pt',reader_path)
        md=importlib.util.module_from_spec(spec);spec.loader.exec_module(md)
        readers={'old':md.Reader(old/'test.pt'),'rich':md.Reader(rich_test)}
        ids=tokens['ids'];n=len(ids)
        def field(name,key):
            r=readers[name];assert len(set(r.meta['ids']))==len(r.meta['ids'])
            lu={v:i for i,v in enumerate(r.meta['ids'])}
            assert set(ids)==set(lu)
            return r.tensor(key)[[lu[v] for v in ids]]
        mask=field('old','modality_mask')
        features=dict(ids=ids,h=test_h,a=field('old','A'),v=field('old','V'),pa=mask[:,1],pv=mask[:,2],
                      A_mean=field('rich','A_mean'),A_local4=field('rich','A_local4'))
        for r in readers.values():
            assert not (set(r.opened)&r.label_storages)
            r.z.close()
        expert=CRBEFExpert(code/'crbef_assets/m3ed').cuda()
        cr_bins={k:[] for k in ('CRBEF','TAV','AV','CR_gate')}
        with torch.no_grad():
            for st in range(0,n,128):
                out=expert({k:v[st:st+128] for k,v in features.items() if k!='ids'})
                for k in cr_bins:cr_bins[k].append(out[k].cpu().numpy())
        cr={k:np.concatenate(v) for k,v in cr_bins.items()};cr['ids']=ids
        cr['P']=np.stack([cr['CRBEF'],cr['CRBEF'],cr['TAV'],cr['TAV']],1)
        np.savez_compressed(root/'CR17_TEST_FEATURES.npz',**features)
        np.savez_compressed(root/'CR17_TEST_PREDICTIONS.npz',**cr)
        save(root/'CR17_TEST_QUALIFICATION.json',dict(status='PASS_ORIGINAL_ADAPTER_VALID32_HIDDEN_REPLAY',
            maximum_hidden_error=herror,test_rows=n,labels_read_by_feature_generator=False,
            adapter_sha256=sha(adapter/'adapter_model.safetensors'),same_weights_as_original_CR17=True,
            upstream_feature_generation='recent4 Qwen3-8B-AWQ with original s43 adapter',
            limitation='head-level valid qualification; original raw train h not recovered'))
        data=Path('/root/autodl-tmp/data/M3ED_textQwen/packed')
        # Every model and recipe was frozen above before reading official test targets.
        ds=open_final_test(data)
        assert [str(v) for v in ds.raw['ids']]==ids.tolist()
        for s in SEEDS:
            status('EVALUATING_FULL_RECOMER_TEST',seed=s)
            complete=ReCoMER.from_bundle(previous/'recomer'/str(s)/'recomer.pt','cuda')
            z=export(complete.mhnou,ds,ids.tolist(),cr)
            fuser=copy.deepcopy(complete.fusion).cpu()
            reports,p=four_way(fuser,z)
            reports.update(TAV=metrics(cr['TAV'],z['y_true']), AV=metrics(cr['AV'],z['y_true']))
            out=root/'full_test'/str(s);out.mkdir(parents=True,exist_ok=False)
            np.savez_compressed(out/'INPUTS.npz',**z)
            np.savez_compressed(out/'PREDICTIONS.npz',ids=ids,y_true=z['y_true'],probabilities=p,
                                MHnoU=z['P'][:,1],cRBEF=z['P'][:,0],equal_weight=z['P'][:,:2].mean(1))
            save(out/'METRICS.json',dict(seed=s,split='test',metrics=reports,model_sha256=sha(previous/'recomer'/str(s)/'recomer.pt'),
                 eta=fuser.cfg.eta,cRBEF_seed_fixed=17,test_selected=False))
            # Live complete-feature forward, excluding upstream encoders and batch construction.
            complete.fusion.cuda();latencies=[];live=[]
            with torch.no_grad():
                for st in range(0,n,128):
                    b,_=ds.batch(range(st,min(st+128,n)));b={k:v.cuda() for k,v in b.items() if k!='history_label'}
                    ef={k:v[st:st+128] for k,v in features.items() if k!='ids'}
                    torch.cuda.synchronize();t=time.perf_counter()
                    result=complete(b,ef)
                    torch.cuda.synchronize();latencies.append(time.perf_counter()-t)
                    live.append(result['fused'].cpu().numpy())
            live=np.concatenate(live)
            changed=int((live.argmax(1)!=p.argmax(1)).sum())
            assert changed==0 and np.abs(live-p).max()<1e-4
            save(out/'COMPUTE.json',dict(parameters_total=sum(v.numel() for v in complete.parameters()),
                 MHnoU_parameters=sum(v.numel() for v in complete.mhnou.parameters()),
                 cRBEF_parameters=sum(v.numel() for v in complete.expert.parameters()),
                 outer_parameters=sum(v.numel() for v in complete.fusion.parameters()),
                 batch_size=128,first_batch_included=True,feature_forward_ms_per_sample=1000*sum(latencies)/n,
                 latency_scope='pooled-feature forward including 2 MH passes and expert heads; upstream Qwen/audio encoders excluded',
                 cache_vs_live_max_probability_error=float(abs(live-p).max()),argmax_changes=changed))
            # Stress both corresponding branches consistently. Text encoder h is zeroed,
            # which is deliberately reported as a feature-level stress intervention.
            stress=[]
            for missing in ('T','A','V','TA','TV','AV'):
                probs=[]
                for st in range(0,n,128):
                    stop=min(st+128,n);b,_=ds.batch(range(st,stop));b={k:v.clone().cuda() for k,v in b.items() if k!='history_label'}
                    ef={k:v[st:stop].copy() for k,v in features.items() if k!='ids'}
                    for m in missing:
                        mi='TAV'.index(m);b[m+'_t'].zero_();b[m+'_h'].zero_()
                        b['modality_mask'][:,mi]=0;b['history_modality_mask'][:,:,mi]=0
                        if m=='T':ef['h']*=0
                        if m=='A':
                            for k in ('a','A_mean','A_local4','pa'):ef[k]*=0
                        if m=='V':
                            for k in ('v','pv'):ef[k]*=0
                    probs.append(complete(b,ef)['fused'].cpu().numpy())
                stress.append(dict(missing=missing,metrics=metrics(np.concatenate(probs),z['y_true'])))
            save(out/'FEATURE_STRESS.json',dict(scope=frozen['robustness_scope'],rows=stress))
            del complete
            torch.cuda.empty_cache()
        status('FULL_TEST_COMPLETE_BASELINES_AND_MECHANISMS_NEXT')
        # Four dataset baselines, fixed valid-selected LR for each arm, all seeds.
        baseline_reports=[]
        for q in selection:
            for s,path in zip(SEEDS,q['selected']['runs']):
                dataset=q['dataset'];d=Path('/root/autodl-tmp/data')/dataset/'packed'
                out=root/'baseline_test'/dataset/q['arm']/str(s)
                status('EVALUATING_BASELINE_TEST',dataset=dataset,arm=q['arm'],seed=s)
                model,cfg,_=load_checkpoint(Path(path)/'best.pt',torch.device('cuda'))
                test=open_final_test(d,task=cfg.task);pred=[];y=[]
                with torch.no_grad():
                    for st in range(0,test.n,256):
                        b,yy=test.batch(range(st,min(st+256,test.n)))
                        b={k:v.cuda() for k,v in b.items() if k!='history_label'}
                        pred.append(model(b)['deployed'].cpu().numpy());y.append(yy.numpy())
                pred=np.concatenate(pred);y=np.concatenate(y)
                met=(classification_report(pred,y,cfg.num_classes,cfg.class_names) if cfg.task=='cls' else regression_report(y,pred.reshape(-1)))
                record=dict(dataset=dataset,arm=q['arm'],seed=s,lr=q['selected']['lr'],task=cfg.task,
                      metrics=met,model_sha256=sha(Path(path)/'best.pt'),split='test',scope='MHnoU only, no cross-dataset cRBEF weights')
                save(out/'METRICS.json',record);out.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(out/'PREDICTIONS.npz',ids=np.asarray(test.raw['ids']),labels=y,logits=pred)
                baseline_reports.append(record);save(root/'BASELINE_TEST_SUMMARY.json',baseline_reports)
                del model,test
                if q['arm']=='mhnou_final':
                    run([sys.executable,str(root/'collect_mechanism_metrics.py'),'--code',str(code),'--data',str(d),
                         '--ckpt',str(Path(path)/'best.pt'),'--out',str(root/'mechanisms'/dataset/str(s)),
                         '--split','test','--device','cuda'],f'mechanisms_{dataset}_{s}')
        # Mechanisms for the exact MHnoU branch used by complete ReCoMER.
        for s in SEEDS:
            run([sys.executable,str(root/'collect_mechanism_metrics.py'),'--code',str(code),'--data',str(data),
                 '--ckpt',str(previous/'mhnou'/str(s)/'final/best.pt'),'--out',str(root/'mechanisms'/'ReCoMER_MH_branch'/str(s)),
                 '--split','test','--device','cuda'],f'mechanisms_ReCoMER_branch_{s}')
            f=load_inputs(previous/'inputs'/str(s)/'FIT.npz');c=load_inputs(previous/'inputs'/str(s)/'final_fusion_cal.npz')
            fuser=fit_fuser(f,c,root/'ablations'/'no_relative_C'/str(s),s,no_c=True)
            z=load_inputs(root/'full_test'/str(s)/'INPUTS.npz');z['C']*=0
            reports,p=four_way(fuser,z)
            save(root/'ablations'/'no_relative_C'/str(s)/'TEST_METRICS.json',dict(seed=s,metrics=reports,split='test'))
            np.savez_compressed(root/'ablations'/'no_relative_C'/str(s)/'TEST_PREDICTIONS.npz',ids=z['ids'],probabilities=p,y_true=z['y_true'])
        status('RETRAINING_MATCHED_FINAL_MODEL_ABLATIONS')
        jobs=load(source/'CONTRACTS.json')['jobs']
        common=['--epochs','20','--batch-size','256','--swa-window','3','--patience','8','--loss-patience','0',
                '--min-delta','0.0001','--device','cuda','--utility','shapley','--gate-architecture','evidence',
                '--gate-detach-inputs','--detach-utility-path','--bounded-w','--bounded-lambda','0.3',
                '--history-abstain-variant','utility_softmax','--deploy','closed_loop']
        for arm in ARMS:
            for s in SEEDS:
                banks=[];folder=root/'ablations'/arm/str(s)
                try:
                    for job in jobs:
                        name=job['name'];out=folder/'mhnou'/name
                        cmd=[sys.executable,str(Path(__file__)),'--train-worker',arm,'--data',str(data),'--out',str(out),
                             '--dialogue-manifest',str(previous/'manifests'/(name+'.json')),'--seed',str(s),'--lr','0.0002',*common]
                        if arm=='no_shapley_supervision':cmd+=['--lambda-u','0']
                        if arm=='no_history_abstention':
                            ix=cmd.index('--history-abstain-variant');cmd[ix+1]='none'
                        status('TRAINING_ABLATION',arm=arm,seed=s,fold=name)
                        run(cmd,f'ablation_{arm}_{s}_{name}')
                        # Export in isolated child to avoid leaking process-local overrides.
                        roles=list(job['predictions'])
                        for role in roles:
                            request=dict(code=str(code),arm=arm,checkpoint=str(out/'best.pt'),data=str(data),
                                 ids=job['predictions'][role],expert_cache=str(source/'pairs'/name/(role+'_INPUTS.npz')),
                                 output=str(folder/'inputs'/(name+'_'+role+'.npz')))
                            request_path=folder/(name+'_'+role+'_REQUEST.json');save(request_path,request)
                            run([sys.executable,str(Path(__file__)),'--export-worker',str(request_path)],f'export_{arm}_{s}_{name}_{role}')
                        if name!='final':banks.append(load_inputs(folder/'inputs'/(name+'_fusion_fit.npz')))
                    joined={k:(banks[0][k] if k=='class_names' else np.concatenate([b[k] for b in banks])) for k in banks[0]}
                    assert len(joined['ids'])==9741 and len(set(joined['ids']))==9741
                    np.savez_compressed(folder/'inputs/FIT.npz',**joined)
                    cal=load_inputs(folder/'inputs/final_fusion_cal.npz')
                    fuser=fit_fuser(joined,cal,folder/'fusion',s)
                    request=dict(code=str(code),arm=arm,checkpoint=str(folder/'mhnou/final/best.pt'),data=str(data),
                          expert_cache=str(root/'CR17_TEST_PREDICTIONS.npz'),output=str(folder/'inputs/TEST.npz'),split='test',
                          expected_checkpoint_sha256=sha(folder/'mhnou/final/best.pt'))
                    save(folder/'ARM_AND_TEST_FROZEN.json',dict(arm=arm,description=ARMS[arm],test_request=request,
                         fusion_sha256=sha(folder/'fusion/local_current.pt'),test_selection=False,harness_sha256=sha(Path(__file__))))
                    req=folder/'TEST_REQUEST.json';save(req,request)
                    run([sys.executable,str(Path(__file__)),'--export-worker',str(req)],f'test_export_{arm}_{s}')
                    z=load_inputs(folder/'inputs/TEST.npz');reports,p=four_way(fuser,z)
                    save(folder/'TEST_METRICS.json',dict(arm=arm,description=ARMS[arm],seed=s,split='test',metrics=reports,
                         eta=fuser.cfg.eta,full_system_OOF=False,cRBEF_seed_fixed=17,test_selected=False))
                    np.savez_compressed(folder/'TEST_PREDICTIONS.npz',ids=z['ids'],probabilities=p,y_true=z['y_true'])
                    status('ABLATION_COMPLETE',arm=arm,seed=s)
                except Exception:
                    save(folder/'FAILURE.json',dict(traceback=traceback.format_exc(),retained=True))
                    status('ABLATION_FAILED_PRESERVED',arm=arm,seed=s)
        status('ALL_REQUESTED_AVAILABLE_TESTS_FINISHED')
        save(root/'COMPLETE.json',dict(status='COMPLETE',seconds=time.monotonic()-started,
             official_full_test_seeds=SEEDS,baseline_runs=len(baseline_reports),mechanism_runs=15,
             retrained_ablation_controls=list(ARMS),attempts=attempts,limitations=frozen['limitations']))
    except BaseException:
        status('FAILED',traceback=traceback.format_exc())
        raise
    finally:
        for h in handles:h.close()


def export_worker(request):
    import numpy as np
    import torch
    sys.path.insert(0,request['code'])
    from n6.data import open_split,open_final_test
    from n6.evaluate import load_checkpoint
    from n6.recomer import mhnou_reference_outputs,combine_inputs
    from n6.crbef_expert import M3ED_CLASSES
    install_arm(request['arm']);torch.set_num_threads(2)
    checkpoint=Path(request['checkpoint'])
    if request.get('expected_checkpoint_sha256'):assert sha(checkpoint)==request['expected_checkpoint_sha256']
    model,cfg,_=load_checkpoint(checkpoint,torch.device('cuda'))
    ds=open_final_test(request['data']) if request.get('split')=='test' else open_split(request['data'],'train.pt')
    lut={str(v):i for i,v in enumerate(ds.raw['ids'])}
    with np.load(request['expert_cache'],allow_pickle=False) as z:cr={k:z[k].copy() for k in ('ids','P')}
    ci={str(v):i for i,v in enumerate(cr['ids'])}
    ids=request.get('ids',[str(v) for v in ds.raw['ids']]);bins={k:[] for k in ('P','C','L0','Lh','M')}
    with torch.no_grad():
        for st in range(0,len(ids),128):
            cur=ids[st:st+128];b,_=ds.batch([lut[v] for v in cur]);b={k:v.cuda() for k,v in b.items() if k!='history_label'}
            mh=mhnou_reference_outputs(model,b);cp=torch.from_numpy(cr['P'][[ci[v] for v in cur]]).cuda()
            p,c=combine_inputs(mh,dict(CRBEF=cp[:,0],TAV=cp[:,2]));out=dict(P=p,**mh)
            for k in bins:bins[k].append(out[k].cpu().numpy())
    values={k:np.concatenate(v) for k,v in bins.items()}
    values.update(ids=np.asarray(ids),source=np.asarray([v.split('_')[1] for v in ids]),class_names=np.asarray(M3ED_CLASSES),
                  y_true=ds.raw['label'][[lut[v] for v in ids]].numpy())
    path=Path(request['output']);path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
    np.savez_compressed(path,**values)
    print(json.dumps(dict(status='EXPORTED',arm=request['arm'],rows=len(ids),split=request.get('split','train'))))


if __name__=='__main__':
    if len(sys.argv)>2 and sys.argv[1]=='--train-worker':
        # argv: script --train-worker arm --data ... ; n6.train accepts argv.
        sys.path.insert(0,str(Path(__file__).resolve().parent/'code'))
        install_arm(sys.argv[2])
        from n6.train import main as train_main
        raise SystemExit(train_main(sys.argv[3:]))
    elif len(sys.argv)>2 and sys.argv[1]=='--export-worker':
        export_worker(load(sys.argv[2]))
    else:main()
