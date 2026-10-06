"""Small read-only check of unattended audit dependencies and tensor APIs."""
import torch
import gate_unattended_finish as f
from n6.evaluate import load_checkpoint
from n6.data import open_split
from n6.train_gate import report
from mosei_gate_benchmark import benchmark_metrics

torch.set_num_threads(2)
for tag in ('meld_roberta','chsims'):
    folder = f.EXP/'gate_feature_screen_20261002/runs'/tag/'seed17/evidence_task'
    model,cfg,_ = load_checkpoint(folder/'best.pt',torch.device('cpu'))
    ds = open_split(f.DATA/f.PACKS[tag],'valid.pt',cfg.task)
    b,y = ds.batch(range(16))
    with torch.no_grad():
        out = model(b)
    metric = 'mae' if cfg.task=='reg' else 'weighted_f1'
    print('SMOKE_OK',tag,report(out['deployed'],y,cfg)[metric],tuple(out['utility_mu'].shape))
print('missing_remaining',len(f.missing()))
