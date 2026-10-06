"""Train-fit-only input transforms; no model/source architecture edits.

The original pack is never overwritten. A transformed checkpoint must be
evaluated through this wrapper with its saved transform metadata.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def fit_statistics(data,out):
    from n6.data import PackedDataset
    from n6.datasetspec import spec_for_pack
    ds=PackedDataset(data/'train.pt',task=spec_for_pack(data).task)
    payload={'train_sha256':sha(data/'train.pt'),'fit_split':'train','moments':{}}
    for j,m in enumerate(['T','A','V']):
        present=ds.modality_mask[:,j]>0
        x=ds.raw[m][present].double()
        if x.ndim!=2 or not len(x) or not torch.isfinite(x).all():
            raise RuntimeError('Cannot fit normalization for '+m)
        payload['moments'][m]={'mean':x.mean(0).float(),
                              'scale':x.std(0,unbiased=False).float().clamp_min(1e-5),
                              'fit_rows':int(present.sum())}
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise RuntimeError('Refusing to overwrite normalization asset')
    torch.save(payload,out)
    return payload


def install_transform(payload,mode):
    from n6.data import PackedDataset
    if getattr(PackedDataset,'_trainfit_transform_installed',False):
        raise RuntimeError('Refusing to apply input normalization twice in one process')
    original=PackedDataset.__init__
    def init(self,*args,**kw):
        original(self,*args,**kw)
        for j,m in enumerate(['T','A','V']):
            raw=self.raw[m].float()
            if not torch.isfinite(raw).all():raise RuntimeError('Nonfinite feature '+m)
            mean=payload['moments'][m]['mean'];scale=payload['moments'][m]['scale']
            if len(mean)!=raw.shape[-1]:raise RuntimeError('Feature dimensions differ from fit asset')
            if mode=='train_zscore':x=(raw-mean)/scale
            elif mode=='train_zclip5':x=((raw-mean)/scale).clamp(-5,5)
            elif mode=='train_center_l2':
                x=raw-mean;x=x/x.norm(dim=-1,keepdim=True).clamp_min(1e-7)
            elif mode=='identity':x=raw
            else:raise ValueError(mode)
            present=self.modality_mask[:,j]>0
            self.raw[m]=torch.where(present[:,None],x,torch.zeros_like(x))
    PackedDataset.__init__=init
    PackedDataset._trainfit_transform_installed=True


def audit(data,out):
    from n6.data import PackedDataset
    from n6.datasetspec import spec_for_pack
    spec=spec_for_pack(data);rows={};ids_by_split={};ds_by_split={}
    for split in ['train','valid']:
        ds=PackedDataset(data/(split+'.pt'),task=spec.task)
        ids=[str(x) for x in ds.raw.get('ids',[])]
        ds_by_split[split]=ds;ids_by_split[split]=ids
        r={'n':ds.n,'dims':ds.dims,'duplicate_ids':len(ids)-len(set(ids)),
           'sha256':sha(data/(split+'.pt')),'modalities':{},'history':{}}
        y=ds.raw['label'];r['nonfinite_labels']=int((~torch.isfinite(y)).sum())
        if spec.task=='cls':r['class_counts']=torch.bincount(y,minlength=spec.num_classes).tolist()
        for j,m in enumerate(['T','A','V']):
            a=ds.raw[m].float();keep=ds.modality_mask[:,j]>0;x=a[keep]
            r['modalities'][m]={'nonfinite':int((~torch.isfinite(a)).sum()),'present_n':int(keep.sum()),
                'zero_present_rows':int((x.abs().sum(-1)==0).sum()),
                'mean_absolute_feature':float(x.abs().mean()),
                'median_feature_std':float(x.std(0,unbiased=False).median())}
        hi=ds.raw.get('history_index')
        if hi is not None:
            r['history']['out_of_bounds_positive']=int((hi>=ds.n).sum())
            r['history']['self_links']=int((hi==torch.arange(ds.n)[:,None]).sum())
            r['history']['larger_row_index_not_equivalent_to_future']=int(((hi>torch.arange(ds.n)[:,None])&(hi<ds.n)).sum())
            if ids and all('$_$' in x for x in ids):
                parsed=[x.rsplit('$_$',1) for x in ids];cross=0;future=0;links=0
                for i,slots in enumerate(hi.tolist()):
                    for h in slots:
                        if 0<=h<ds.n:
                            links+=1;cross+=parsed[h][0]!=parsed[i][0]
                            future+=int(parsed[h][1])>=int(parsed[i][1])
                r['history'].update(cross_video=cross,not_past_segment=future,valid_links=links,
                                    chronology_source='Numeric segment suffix in MOSEI ids; not array row order')
        rows[split]=r
    lookup={v:i for i,v in enumerate(ids_by_split['train'])}
    pairs=[(lookup[v],j) for j,v in enumerate(ids_by_split['valid']) if v in lookup]
    exact_all=0
    for i,j in pairs:
        if all(torch.equal(ds_by_split['train'].raw[m][i],ds_by_split['valid'].raw[m][j])
               for m in ['T','A','V']):exact_all+=1
    report={'dataset_id':spec.dataset_id,'splits':rows,'test_read':False,
            'cross_split_raw_id_collisions':len(pairs),'identical_TAV_for_colliding_ids':exact_all,
            'id_caveat':'MELD identifiers are split-local; identifier collision alone is not duplicate sample or leakage evidence.',
            'chronology_caveat':'Larger array row indices do not establish future leakage; inspect dataset IDs/time metadata.',
            'rows_removed':0,'labels_changed':False,'splits_changed':False,
            'upstream_feature_training_provenance':'Not established by a packed-feature numerical audit',
            'model_architecture_changed':False}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'status':'AUDIT_COMPLETE','dataset':spec.dataset_id,'out':str(out)}),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--code',type=Path,required=True)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--action',choices=['audit','fit','train','evaluate'],required=True)
    ap.add_argument('--moments',type=Path)
    ap.add_argument('--mode',choices=['identity','train_zscore','train_zclip5','train_center_l2'],default='identity')
    ap.add_argument('--out',type=Path)
    args,extra=ap.parse_known_args()
    sys.path.insert(0,str(args.code.resolve()));torch.set_num_threads(2)
    if extra and extra[0]=='--':extra=extra[1:]
    if args.action=='audit':
        if extra or args.out is None:raise RuntimeError('Audit requires --out and no trainer arguments')
        return audit(args.data,args.out)
    if args.moments is None:raise RuntimeError('Normalization asset path required')
    if args.action=='fit':return fit_statistics(args.data,args.moments)
    payload=torch.load(args.moments,map_location='cpu',weights_only=True)
    if payload['fit_split']!='train' or sha(args.data/'train.pt')!=payload['train_sha256']:
        raise RuntimeError('Fit asset is not bound to this train pack')
    install_transform(payload,args.mode)
    if args.action=='train':
        if args.out is None:raise RuntimeError('Training output directory required')
        from n6.train import main as train
        result=train(['--data',str(args.data),'--out',str(args.out)]+extra)
        metadata={'input_transform':args.mode,'moments_path':str(args.moments.resolve()),
                  'moments_sha256':sha(args.moments),'train_sha256':payload['train_sha256'],
                  'fit_split':'train','labels_changed':False,'rows_removed':0,'splits_changed':False,
                  'required_evaluator':'data_pipeline_probe.py --action evaluate',
                  'model_architecture_changed':False}
        (args.out/'DATA_PIPELINE.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        return result
    from n6.evaluate import main as evaluate
    if '--ckpt' not in extra:raise RuntimeError('Checkpoint required')
    ckpt=Path(extra[extra.index('--ckpt')+1]);metadata=json.loads((ckpt.parent/'DATA_PIPELINE.json').read_text())
    if metadata['moments_sha256']!=sha(args.moments) or metadata['input_transform']!=args.mode:
        raise RuntimeError('Evaluator transformation differs from training')
    argv=['--data',str(args.data)]+extra
    if args.out is not None:argv+=['--out',str(args.out)]
    return evaluate(argv)


if __name__=='__main__':main()
