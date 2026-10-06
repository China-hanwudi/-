"""Additional CPU-only implementation/data audits; no parameter updates.

The gate formula interventions are post-hoc descriptive sensitivity checks,
not new confirmatory trials or selected improvements.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def main(root):
    previous=Path('/root/recomer_complete_20261004_side')
    sys.path.insert(0,str(root/'code'))
    from n6.data import open_split,open_final_test
    from n6.recomer import ReCoMER
    from n6.metrics import classification_report
    from n6.crbef_expert import M3ED_CLASSES
    torch.set_num_threads(2)
    def save(name,z):(root/name).write_text(json.dumps(z,indent=2),encoding='utf-8')
    # Complete pooled-feature metadata and disjoint-ID audit on all used data.
    records=[]
    for name,task in [('M3ED_textQwen','cls'),('MELD','cls'),('IEMOCAP','cls'),('MOSEI_full','reg')]:
        folder=Path('/root/autodl-tmp/data')/name/'packed';parts={};ids={};fingerprints={}
        for split in ('train','valid','test'):
            ds=open_final_test(folder,task) if split=='test' else open_split(folder,split+'.pt',task)
            labels=ds.raw['label'].numpy();sampleids=[str(v) for v in ds.raw['ids']]
            ids[split]=set(sampleids);hi=ds.raw.get('history_index')
            fingerprints[split]={}
            for i,identifier in enumerate(sampleids):
                h=hashlib.sha256()
                for m in 'TAV':h.update(ds.raw[m][i].contiguous().numpy().tobytes())
                fingerprints[split][identifier]=h.hexdigest()
            parts[split]=dict(samples=ds.n,input_dims=ds.dims,history_k=ds.k,unique_ids=len(ids[split]),
               sha256=sha(folder/(split+'.pt')),modal_presence_counts=ds.modality_mask.sum(0).tolist(),
               label_counts=np.bincount(labels.astype(np.int64)).tolist() if task=='cls' else None,
               label_range=[float(labels.min()),float(labels.max())],finite_features=all(bool(torch.isfinite(ds.raw[m]).all()) for m in 'TAV'))
            if hi is not None:
                rr,ss=(hi>=0).nonzero(as_tuple=True);cc=hi[rr,ss]
                parts[split].update(history_links=len(rr),invalid_history_indices=int((cc>=ds.n).sum()),
                    history_indices_to_later_row=int((cc>=rr).sum()),
                    history_order_check_scope='index order only; chronology requires dataset-specific timestamps')
            del ds
        overlap={f'{a}_{b}':len(ids[a]&ids[b]) for a,b in [('train','valid'),('train','test'),('valid','test')]}
        matching_shared_ids={f'{a}_{b}':sum(fingerprints[a][k]==fingerprints[b][k] for k in ids[a]&ids[b])
             for a,b in [('train','valid'),('train','test'),('valid','test')]}
        assert all(p['samples']==p['unique_ids'] and p['finite_features'] for p in parts.values())
        records.append(dict(dataset=name,task=task,splits=parts,split_id_overlap=overlap,
             shared_ids_with_identical_T_A_V=matching_shared_ids,
             identity_scope='MELD dialogue/utterance IDs are split-local; identifier collisions alone do not establish leakage. Identical feature triples are descriptive, not proof of identical raw samples.'))
    save('DATA_AUDIT.json',dict(scope='metadata and features; official test targets read after plan freeze; no fitting',datasets=records))
    with np.load(root/'CR17_TEST_PREDICTIONS.npz',allow_pickle=False) as z:cr=dict(z)
    test=open_final_test('/root/autodl-tmp/data/M3ED_textQwen/packed')
    assert np.array_equal(cr['ids'],np.asarray(test.raw['ids']))
    y=test.raw['label'].numpy();expert=ReCoMER.from_bundle(previous/'recomer/43/recomer.pt').expert
    p=torch.from_numpy(cr['TAV']);av=torch.from_numpy(cr['AV']);g=torch.from_numpy(cr['CR_gate'])
    evidence=(av+1e-9).log()-(expert.prior+1e-9).log()
    probes={}
    for name,logp in {
        'TAV_only_gate0':(p+1e-9).log(),
        'constant_gate_half':(p+1e-9).log()+.5*evidence,
        'constant_gate_one':(p+1e-9).log()+evidence,
        'learned_gate_without_prior_correction':(p+1e-9).log()+g*(av+1e-9).log(),
        'TAV_AV_probability_average':((p+av)/2).clamp_min(1e-9).log(),
        'original_CR17':(p+1e-9).log()+g*evidence,
    }.items():
        probes[name]=classification_report(logp.numpy(),y,7,M3ED_CLASSES)
    save('CR_GATE_FORMULA_SENSITIVITY.json',dict(scope='post-hoc frozen formula interventions, no refitting or selection',
           probes=probes,cRBEF_seed_fixed=17,original_weights_preserved=True))
    # Save empty-candidate mass and history-state evidence missing from older diagnostics.
    for seed in (43,47,59):
        full=ReCoMER.from_bundle(previous/'recomer'/str(seed)/'recomer.pt');bins={}
        with torch.no_grad():
            for st in range(0,test.n,128):
                b,_=test.batch(range(st,min(st+128,test.n)));b.pop('history_label')
                out=full.mhnou(b)
                for k in ('abstain','history_gate_features','history_gate_logit','has_history'):
                    v=out[k]
                    if v is not None:bins.setdefault(k,[]).append(v.numpy())
                bins.setdefault('speaker_same',[]).append(b['speaker_same'].numpy())
        arrays={k:np.concatenate(v) for k,v in bins.items()};arrays['ids']=np.asarray(test.raw['ids'])
        target=root/'mechanisms/ReCoMER_MH_branch'/str(seed)
        np.savez_compressed(target/'EMPTY_CANDIDATE_AND_HISTORY_STATE.npz',**arrays)
        mask=arrays['has_history'].astype(bool)
        save(f'EMPTY_CANDIDATE_SEED{seed}.json',dict(scope='fixed checkpoint official-test description',seed=seed,
             eligible_history_samples=int(mask.sum()),mean_null_mass_by_modality=arrays['abstain'][mask].mean(0).tolist(),
             minimum_null_mass_by_modality=arrays['abstain'][mask].min(0).tolist(),
             maximum_null_mass_by_modality=arrays['abstain'][mask].max(0).tolist()))
    # Explicitly verify which module the external expert can influence.
    valid=open_split('/root/autodl-tmp/data/M3ED_textQwen/packed','valid.pt')
    with np.load('/root/autodl-tmp/fusion_research_automation_20261003/round032_same_cr_valid/full_v2/features.npz',allow_pickle=False) as z:
        f={k:z[k][:32].copy() for k in ('h','a','v','pa','pv')}
    rich=torch.load('/root/autodl-tmp/multimodal_newbench_20260912/runs/m3ed_e2v_plus_audio_003/valid_features.pt',map_location='cpu',weights_only=True)
    lookup={str(v):i for i,v in enumerate(rich['ids'])};v_ids=[str(v) for v in valid.raw['ids'][:32]]
    f.update(A_mean=rich['A_mean'][[lookup[v] for v in v_ids]],A_local4=rich['A_local4'][[lookup[v] for v in v_ids]])
    altered={k:(v.flip(0) if torch.is_tensor(v) else v[::-1].copy()) for k,v in f.items()}
    b,_=valid.batch(range(32));b.pop('history_label');flow=[]
    with torch.no_grad():
        for seed in (43,47,59):
            full=ReCoMER.from_bundle(previous/'recomer'/str(seed)/'recomer.pt')
            a=full(b,f);c=full(b,altered)
            flow.append(dict(seed=seed,MH_probability_difference=float((a['MHnoU']-c['MHnoU']).abs().max()),
                 modality_relative_C_difference=float((a['C']-c['C']).abs().max()),
                 CR_probability_difference=float((a['CRBEF']-c['CRBEF']).abs().max()),
                 full_probability_difference=float((a['fused']-c['fused']).abs().max())))
    save('EXPERT_TO_MHNOU_INFORMATION_FLOW.json',dict(scope='valid-only feature intervention, no training',rows=flow,
       interpretation='External cRBEF affects outer class-probability fusion. It is not an input to MHnoU contribution router and does not directly change its modal weights.'))
    # Preserve production identity evidence after the test/model probes.
    plan=json.loads((root/'PLAN.json').read_text())
    checks={p:sha(p)==h for p,h in plan['asset_sha256'].items()}
    codechecks={p:sha(root/'code'/p)==h for p,h in plan['code_sha256'].items()}
    assert all(checks.values()) and all(codechecks.values())
    save('PRODUCTION_IDENTITY_AUDIT.json',dict(frozen_assets_unchanged=checks,production_code_unchanged=codechecks,
          paused_workflow_current=json.loads(Path('/root/autodl-tmp/fusion_research_automation_20261003/CURRENT.json').read_text()),
          audit_time=time.time(),parameter_updates_in_this_audit=0))
    print(json.dumps(dict(status='ADDITIONAL_AUDITS_COMPLETE',datasets=len(records),gate_probes=len(probes),seeds=3)))


if __name__=='__main__':main(Path(sys.argv[1]))
