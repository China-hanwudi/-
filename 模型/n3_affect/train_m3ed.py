"""Train/evaluate the active M3ED classification path.

Only packed train.pt and valid.pt are read.  The sealed test split is never
opened by this entry point.  Each seed has an isolated output directory.
"""
from __future__ import annotations
import argparse, hashlib, json, random, sys
from pathlib import Path
import numpy as np
import torch
from .m3ed_dataset import M3EDPackedDataset
from .config import N3TrainConfig
from .model import N3EmotionModel
from .losses import n3_total_loss

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def f1(labels, preds, c=7):
    cm=np.zeros((c,c),dtype=np.int64)
    for y,p in zip(labels,preds):
        if 0<=int(y)<c and 0<=int(p)<c: cm[int(y),int(p)]+=1
    tp=np.diag(cm).astype(float); sup=cm.sum(1).astype(float); ps=cm.sum(0).astype(float)
    pr=np.divide(tp,ps,out=np.zeros_like(tp),where=ps>0); rc=np.divide(tp,sup,out=np.zeros_like(tp),where=sup>0)
    z=np.divide(2*pr*rc,pr+rc,out=np.zeros_like(tp),where=(pr+rc)>0)
    return float(z.mean()), float((z*sup).sum()/max(sup.sum(),1))

@torch.no_grad()
def evaluate(model, ds, device, bs, max_samples=None):
    model.eval(); ys=[]; ps=[]; loss=0.; nstep=0
    total = ds.n if max_samples is None else min(ds.n, int(max_samples))
    for s in range(0, total, bs):
        b,y=ds.batch(range(s,min(s+bs,total))); b={k:v.to(device) for k,v in b.items()}; y=y.to(device)
        out=model(b, route_mode="hard-safe"); loss+=float(n3_total_loss(out,y,model.cfg)["loss"]); nstep+=1
        ys.extend(y.cpu().tolist()); ps.extend(out["logits"].argmax(-1).cpu().tolist())
    ma, wf=f1(ys,ps,model.cfg.num_classes); return {"loss":loss/max(nstep,1),"accuracy":float(np.mean(np.asarray(ys)==np.asarray(ps))),"macro_f1":ma,"weighted_f1":wf,"n":len(ys)}

def sha(path):
    h=hashlib.sha256();
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(1<<20),b''): h.update(x)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1]); ap.add_argument('--data',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--seed',type=int,default=17); ap.add_argument('--epochs',type=int,default=20); ap.add_argument('--batch-size',type=int,default=64); ap.add_argument('--smoke',type=int,default=0); args=ap.parse_args()
    seed_all(args.seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); train=M3EDPackedDataset(args.data/'train.pt'); valid=M3EDPackedDataset(args.data/'valid.pt')
    cfg=N3TrainConfig(text_dim=train.raw['T'].shape[-1],audio_dim=train.raw['A'].shape[-1],video_dim=train.raw['V'].shape[-1],num_classes=7,emotion_label_order=tuple(f'class_{i}' for i in range(7)),text_tower='composer_n3',dropout=0.1,batch_size=args.batch_size,max_epochs=args.epochs,seed=args.seed,counterfactual_loss_weight=0.0)
    model=N3EmotionModel(cfg).to(device); opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay); args.out.mkdir(parents=True,exist_ok=True)
    meta={'dataset':'M3ED','seed':args.seed,'device':str(device),'train_samples':train.n,'valid_samples':valid.n,'dims':{'T':int(cfg.text_dim),'A':int(cfg.audio_dim),'V':int(cfg.video_dim)},'train_sha256':sha(train.path),'valid_sha256':sha(valid.path),'test_read':False,'history_contract':'oldest_to_newest_right_aligned_K3'}
    (args.out/'RUN_METADATA.json').write_text(json.dumps(meta,indent=2),encoding='utf-8'); best=None; history=[]
    epochs=1 if args.smoke else args.epochs
    for ep in range(epochs):
        model.train(); model.current_epoch=ep; perm=torch.randperm(train.n); running=0.; steps=0
        for st in range(0,train.n,args.batch_size):
            if args.smoke and steps>=args.smoke: break
            ix=perm[st:min(st+args.batch_size,train.n)].tolist(); b,y=train.batch(ix); b={k:v.to(device) for k,v in b.items()}; y=y.to(device)
            out=model(b,route_mode='hard-safe'); loss=n3_total_loss(out,y,cfg)['loss']; opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip); opt.step(); running+=float(loss.detach()); steps+=1
        m=evaluate(model,valid,device,args.batch_size, max_samples=(args.batch_size * 2 if args.smoke else None)); row={'epoch':ep,'train_loss':running/max(steps,1),**m}; history.append(row); print(json.dumps(row),flush=True)
        if best is None or (m['weighted_f1'],m['macro_f1'],-m['loss'])>(best['weighted_f1'],best['macro_f1'],-best['loss']):
            best=dict(m); torch.save({'model':model.state_dict(),'cfg':cfg.to_dict(),'seed':args.seed,'epoch':ep,'valid':m},args.out/'best.pt')
    (args.out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    if not (args.out/'best.pt').is_file(): raise RuntimeError('missing best.pt')
    final={'dataset':'M3ED','seed':args.seed,'best_valid':best,'status':'SMOKE_PASS' if args.smoke else 'TRAIN_COMPLETE','test_read':False}
    (args.out/'FINAL_RESULT.json').write_text(json.dumps(final,indent=2),encoding='utf-8'); print('FINAL_RESULT',json.dumps(final),flush=True)
if __name__=='__main__': main()
