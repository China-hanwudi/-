"""CPU-only, post-hoc full-model feature-intervention Shapley audit.

This game removes a modality's CURRENT AND HISTORY features in both branches.
Text removal zeros the original Qwen hidden feature; it does not re-encode
raw prompts. It is different from the training MHnoU weighted-token game.
No target enters a forward; labels only score the resulting probabilities.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import itertools
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch


def main(root):
    sys.path.insert(0,str(root/'code'))
    from n6.data import open_final_test
    from n6.recomer import ReCoMER
    torch.set_num_threads(2)
    started=time.monotonic();out=root/'full_feature_shapley';out.mkdir(exist_ok=False)
    models=Path('/root/recomer_complete_20261004_side/recomer')
    ds=open_final_test('/root/autodl-tmp/data/M3ED_textQwen/packed')
    with np.load(root/'CR17_TEST_FEATURES.npz',allow_pickle=False) as z:features=dict(z)
    ids=np.asarray(ds.raw['ids']);assert np.array_equal(features['ids'],ids)
    subsets=[s for r in range(4) for s in itertools.combinations(range(3),r)]
    roles=('MHnoU','cRBEF','ReCoMER')
    plan=dict(scope='post-hoc fixed-model feature interventions; not original training teacher or raw-missing-modality benchmark',
         seeds=[43,47,59],subsets=[list(s) for s in subsets],utility='log probability of true class',
         baseline='actual all-zero/masked feature forward, not fixed utility 0',
         labels_used_by_forward=False,parameter_updates=0,mask_current_and_history=True,
         text_absence='zero precomputed Qwen hidden representation, no raw prompt re-encoding',
         expert_seed_fixed=17,select_models_from_results=False)
    (out/'PLAN.json').write_text(json.dumps(plan,indent=2))
    summaries=[]
    with torch.no_grad():
        for seed in plan['seeds']:
            model=ReCoMER.from_bundle(models/str(seed)/'recomer.pt');utility={role:{} for role in roles}
            for subset in subsets:
                bins={role:[] for role in roles}
                for st in range(0,ds.n,128):
                    stop=min(st+128,ds.n);b,y=ds.batch(range(st,stop))
                    b={k:v.clone() for k,v in b.items() if k!='history_label'}
                    ef={k:v[st:stop].copy() for k,v in features.items() if k!='ids'}
                    for i,m in enumerate('TAV'):
                        if i in subset:continue
                        b[m+'_t'].zero_();b[m+'_h'].zero_();b['modality_mask'][:,i]=0;b['history_modality_mask'][:,:,i]=0
                        if m=='T':ef['h']*=0
                        if m=='A':
                            for k in ('a','A_mean','A_local4','pa'):ef[k]*=0
                        if m=='V':
                            for k in ('v','pv'):ef[k]*=0
                    pred=model(b,ef)
                    for role,key in [('MHnoU','MHnoU'),('cRBEF','CRBEF'),('ReCoMER','fused')]:
                        p=pred[key].numpy();bins[role].append(np.log(np.maximum(p[np.arange(stop-st),y.numpy()],1e-9)))
                for role in roles:utility[role][subset]=np.concatenate(bins[role]).astype(np.float64)
                (out/'PROGRESS.json').write_text(json.dumps(dict(seed=seed,subset=list(subset),seconds=time.monotonic()-started)))
            phis={}
            for role in roles:
                phi=np.zeros((ds.n,3),np.float64)
                for i in range(3):
                    others=[j for j in range(3) if j!=i]
                    for r in range(3):
                        for s in itertools.combinations(others,r):
                            w=math.factorial(r)*math.factorial(2-r)/6
                            phi[:,i]+=w*(utility[role][tuple(sorted((*s,i)))]-utility[role][s])
                err=float(np.max(np.abs(phi.sum(1)-(utility[role][(0,1,2)]-utility[role][()]))))
                assert err<=1e-6
                phis[role]=phi
            with np.load(root/'mechanisms/ReCoMER_MH_branch'/str(seed)/'SAMPLE_DIAGNOSTICS.npz',allow_pickle=False) as z:diag=dict(z)
            eligible=(diag['modality_mask']>0).all(1)
            row=dict(seed=seed,n=int(eligible.sum()),alignment_to_MH_weights={},dominant_modality_counts={})
            for role in roles:
                teacher=phis[role][eligible];weight=diag['weights'][eligible]
                row['alignment_to_MH_weights'][role]=dict(top1=float((teacher.argmax(1)==weight.argmax(1)).mean()),
                    pairwise=float(np.stack([np.sign(teacher[:,i]-teacher[:,j])==np.sign(weight[:,i]-weight[:,j])
                                            for i,j in [(0,1),(0,2),(1,2)]],1).mean()))
                row['dominant_modality_counts'][role]=np.bincount(teacher.argmax(1),minlength=3).tolist()
            np.savez_compressed(out/f'SEED{seed}.npz',ids=ids,labels=ds.raw['label'].numpy(),
                weights=diag['weights'],phi_MHnoU=phis['MHnoU'],phi_cRBEF=phis['cRBEF'],phi_ReCoMER=phis['ReCoMER'],
                utility_MHnoU=np.stack([utility['MHnoU'][s] for s in subsets],1),
                utility_cRBEF=np.stack([utility['cRBEF'][s] for s in subsets],1),
                utility_ReCoMER=np.stack([utility['ReCoMER'][s] for s in subsets],1))
            summaries.append(row)
            (out/'SUMMARY.json').write_text(json.dumps(dict(plan=plan,seeds=summaries,seconds=time.monotonic()-started),indent=2))
    (out/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',seeds=3,subset_forwards=24,seconds=time.monotonic()-started),indent=2))
    print(json.dumps(dict(status='COMPLETE_FEATURE_SHAPLEY',seeds=summaries,seconds=time.monotonic()-started)))


if __name__=='__main__':main(Path(sys.argv[1]))
