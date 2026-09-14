"""Train CMU-MOSEI scalar sentiment on train/valid packed splits only."""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
import numpy as np, torch
from .mosei_dataset import MOSEIPackedDataset
from .regression_config import N3RegressionConfig
from .regression_model import N3SentimentModel
from .regression_losses import n3_regression_loss

def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
@torch.no_grad()
def evaluate(model,ds,device,bs,max_samples=None):
    model.eval(); p=[]; y=[]
    total = ds.n if max_samples is None else min(ds.n, int(max_samples))
    for st in range(0,total,bs):
        b,t=ds.batch(range(st,min(st+bs,total))); b={k:v.to(device) for k,v in b.items()}; o=model(b,'hard-safe'); p.extend(o['prediction'].squeeze(-1).cpu().tolist()); y.extend(t.tolist())
    p=np.asarray(p,float); y=np.asarray(y,float); mae=float(np.mean(np.abs(p-y))); mse=float(np.mean((p-y)**2)); corr=float(np.corrcoef(p,y)[0,1]) if np.std(p)>0 and np.std(y)>0 else 0.; nz=(y!=0); f1n=0.
    if nz.any():
        pp=p!=0; tp=((pp)&nz).sum(); fp=((pp)&(~nz)).sum(); fn=((~pp)&nz).sum(); f1n=float(2*tp/max(2*tp+fp+fn,1))
    return {'MAE':mae,'MSE':mse,'Pearson':corr,'F1_nonzero':f1n,'n':len(y)}
def sha(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for x in iter(lambda:f.read(1<<20),b''): h.update(x)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--seed',type=int,default=17); ap.add_argument('--epochs',type=int,default=20); ap.add_argument('--batch-size',type=int,default=64); ap.add_argument('--smoke',type=int,default=0); args=ap.parse_args()
    seed_all(args.seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); tr=MOSEIPackedDataset(args.data/'train.pt'); va=MOSEIPackedDataset(args.data/'valid.pt')
    cfg=N3RegressionConfig(text_dim=tr.raw['T'].shape[-1],audio_dim=tr.raw['A'].shape[-1],video_dim=tr.raw['V'].shape[-1],target_scale=3.0,task='mosei_sentiment_regression',batch_size=args.batch_size,max_epochs=args.epochs,seed=args.seed,counterfactual_loss_weight=0.20,unimodal_loss_weight=0.0)
    m=N3SentimentModel(cfg).to(device); opt=torch.optim.AdamW(m.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay); args.out.mkdir(parents=True,exist_ok=True)
    meta={'dataset':'CMU-MOSEI','seed':args.seed,'device':str(device),'train_samples':tr.n,'valid_samples':va.n,'dims':{'T':cfg.text_dim,'A':cfg.audio_dim,'V':cfg.video_dim},'target':'raw sentiment in [-3,3], model normalizes y/3 and reports 3*u','train_sha256':sha(tr.path),'valid_sha256':sha(va.path),'test_read':False,'history_contract':'oldest_to_newest_right_aligned_K3'}; (args.out/'RUN_METADATA.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    best=None; hist=[]; epochs=1 if args.smoke else args.epochs
    for ep in range(epochs):
        m.train(); m.current_epoch=ep; perm=torch.randperm(tr.n); run=0.; steps=0
        for st in range(0,tr.n,args.batch_size):
            if args.smoke and steps>=args.smoke: break
            ix=perm[st:min(st+args.batch_size,tr.n)].tolist(); b,y=tr.batch(ix); b={k:v.to(device) for k,v in b.items()}; y=y.to(device); o=m(b,'hard-safe'); o['cf_measured_targets']=m.measure_counterfactual_utility(b,y); z=n3_regression_loss(o,y,cfg)['loss']; opt.zero_grad(set_to_none=True); z.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),cfg.grad_clip); opt.step(); run+=float(z.detach()); steps+=1
        met=evaluate(m,va,device,args.batch_size,max_samples=(args.batch_size * 2 if args.smoke else None)); row={'epoch':ep,'train_loss':run/max(steps,1),**met}; hist.append(row); print(json.dumps(row),flush=True)
        if best is None or met['MAE']<best['MAE']: best=dict(met); torch.save({'model':m.state_dict(),'cfg':cfg.to_dict(),'seed':args.seed,'epoch':ep,'valid':met},args.out/'best.pt')
    (args.out/'history.json').write_text(json.dumps(hist,indent=2),encoding='utf-8'); final={'dataset':'CMU-MOSEI','seed':args.seed,'best_valid':best,'status':'SMOKE_PASS' if args.smoke else 'TRAIN_COMPLETE','test_read':False}; (args.out/'FINAL_RESULT.json').write_text(json.dumps(final,indent=2),encoding='utf-8'); print('FINAL_RESULT',json.dumps(final),flush=True)
if __name__=='__main__': main()
