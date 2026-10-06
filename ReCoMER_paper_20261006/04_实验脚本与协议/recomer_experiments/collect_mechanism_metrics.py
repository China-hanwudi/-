"""Read-only checkpoint diagnostics. Labels are used for analysis, never tuning.

Outputs are counterfactual input probes of one frozen checkpoint, not retrained
ablations or evidence that the historical labels were available at deployment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--code', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--split', choices=['valid', 'test'], default='valid')
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--max-samples', type=int, default=0)
    args = ap.parse_args()
    sys.path.insert(0, str(args.code.resolve()))
    from n6.data import open_split, open_final_test
    from n6.evaluate import load_checkpoint
    from n6.metrics import classification_report, regression_report
    from n6.datasetspec import class_names_for

    args.out.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.set_num_threads(2)
    model, cfg, ck = load_checkpoint(args.ckpt, device)
    model.eval()
    pipeline_path=args.ckpt.parent/'DATA_PIPELINE.json'
    pipeline=None
    if pipeline_path.exists():
        pipeline=json.loads(pipeline_path.read_text())
        sys.path.insert(0,str(Path(__file__).resolve().parent))
        from data_pipeline_probe import install_transform,sha
        moments=Path(pipeline['moments_path'])
        if sha(moments)!=pipeline['moments_sha256'] or sha(args.data/'train.pt')!=pipeline['train_sha256']:
            raise RuntimeError('Diagnostics do not match saved train-only transform')
        install_transform(torch.load(moments,map_location='cpu',weights_only=True),pipeline['input_transform'])
    ds = (open_split(args.data, 'valid.pt', task=cfg.task) if args.split == 'valid'
          else open_final_test(args.data, task=cfg.task))
    total = min(ds.n, args.max_samples) if args.max_samples else ds.n
    bins = {}
    timers = {'clean_seconds': 0.0, 'shapley_seconds': 0.0}

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def append(key, value):
        if torch.is_tensor(value):
            value = value.detach().float().cpu().numpy()
        bins.setdefault(key, []).append(np.asarray(value))

    def scalar(out, key, size):
        value = out.get(key)
        return (torch.full((size,), float('nan'), device=device) if value is None
                else value.reshape(size, -1).mean(1))

    def loss(logits, y):
        return (F.cross_entropy(logits, y, reduction='none') if cfg.task == 'cls'
                else (logits.squeeze(-1) - y).square())

    # Warm-up excluded from inference latency; no labels needed for this forward.
    b, _ = ds.batch(range(min(8, total)))
    with torch.no_grad():
        model({k: v.to(device) for k, v in b.items()})
    sync()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    began = time.perf_counter()
    with torch.no_grad():
        for start in range(0, total, args.batch_size):
            stop = min(start + args.batch_size, total)
            b, y = ds.batch(range(start, stop))
            b = {k: v.to(device) for k, v in b.items()}
            y = y.to(device)
            sync(); t = time.perf_counter(); clean = model(b); sync()
            timers['clean_seconds'] += time.perf_counter()-t
            off = model(b, history_override=0)
            on = model(b, history_override=1)
            w = clean['weights'].detach() if cfg.deploy == 'closed_loop' else None
            sync(); t = time.perf_counter()
            phi = model.measure_shapley(b, y, standardize=False, token_weight=w)
            sync(); timers['shapley_seconds'] += time.perf_counter()-t
            for key, val in {'labels': y, 'deployed': clean['deployed'],
                             'history_off_logits': off['deployed'],
                             'history_on_logits': on['deployed'],
                             'weights': clean['weights'], 'solo_logits': clean['solo_stack'],
                             'shapley_raw': phi, 'modality_mask': b['modality_mask'],
                             'loss_clean': loss(clean['deployed'], y),
                             'loss_history_off': loss(off['deployed'], y),
                             'loss_history_on': loss(on['deployed'], y),
                             'token_mask': clean['token_mask'],
                             'token_mask_off': off['token_mask'],
                             'token_mask_on': on['token_mask'],
                             'history_mask': b['history_mask']}.items():
                append(key, val)
            append('utility_mu', clean['utility_mu'] if clean['utility_mu'] is not None
                   else torch.full_like(phi, float('nan')))
            for key in ['history_gate_hard', 'history_gate_prob', 'history_gate_applied']:
                append(key, scalar(clean, key, stop-start))
            for key in ['cur_embs']:
                delta = torch.stack([clean[key][m]-off[key][m] for m in ['T','A','V']], 1)
                append('history_representation_delta_norm', delta.norm(dim=-1))
            # Dataset adapter stores canonical oldest-to-newest history labels.
            # Last valid slot defines a ground-truth analysis-only transition proxy.
            group = torch.full((stop-start,), -1, device=device, dtype=torch.long)
            if cfg.task == 'cls' and b['history_mask'].shape[1]:
                available = b['history_mask'].sum(1) > 0
                k = b['history_mask'].shape[1]
                last = k-1-b['history_mask'].flip(1).argmax(1)
                previous = b['history_label'].gather(1, last[:, None]).squeeze(1)
                group[available] = (previous[available] != y[available]).long()
            append('transition_group', group)
    arrays = {key: np.concatenate(values) for key, values in bins.items()}
    raw_ids = ds.raw.get('ids')
    ids = ([str(v) for v in raw_ids[:total]] if raw_ids is not None
           else [f'{args.split}:{i}' for i in range(total)])
    arrays['ids'] = np.asarray(ids, dtype=str)
    classes = class_names_for(cfg, ds.path) or []
    arrays['class_names'] = np.asarray(classes, dtype=str)
    arrays['labels'] = arrays['labels'].astype(np.int64 if cfg.task == 'cls' else np.float32)
    np.savez_compressed(args.out/'SAMPLE_DIAGNOSTICS.npz', **arrays)
    report_fn = (lambda logits, y: classification_report(logits, y, cfg.num_classes, classes)
                 if cfg.task == 'cls' else regression_report(y, logits.reshape(-1)))
    groups = {}
    for group, name in [(-1, 'no_history_or_not_applicable'), (0, 'label_continuation'), (1, 'label_flip')]:
        keep = arrays['transition_group'] == group
        if keep.any():
            groups[name] = dict(n=int(keep.sum()), metrics=report_fn(arrays['deployed'][keep], arrays['labels'][keep]),
                               mean_history_on_minus_off_loss=float((arrays['loss_history_on'][keep]-arrays['loss_history_off'][keep]).mean()))
    mu = arrays['utility_mu']
    phi = arrays['shapley_raw']
    eligible = np.isfinite(mu).all(1) & (arrays['modality_mask'] > 0).all(1)
    alignment = None
    if eligible.any():
        alignment = dict(n=int(eligible.sum()),
            top1_agreement=float((mu[eligible].argmax(1) == phi[eligible].argmax(1)).mean()),
            pairwise_order_agreement=float(np.stack([
                np.sign(mu[eligible,i]-mu[eligible,j]) == np.sign(phi[eligible,i]-phi[eligible,j])
                for i,j in [(0,1),(0,2),(1,2)]],1).mean()))
    improvement = arrays['loss_history_off']-arrays['loss_history_on']
    hard = arrays['history_gate_hard']
    eligible = np.isfinite(hard) & (arrays['history_mask'].sum(1)>0)
    gate_accuracy = None
    if eligible.any():
        gate_accuracy = dict(n=int(eligible.sum()),
            helpful_rejected=int(((improvement>1e-6)&(hard<0.5)&eligible).sum()),
            harmful_accepted=int(((improvement<-1e-6)&(hard>=0.5)&eligible).sum()),
            helpful_accepted=int(((improvement>1e-6)&(hard>=0.5)&eligible).sum()),
            harmful_rejected=int(((improvement<-1e-6)&(hard<0.5)&eligible).sum()))
    training_result_path=args.ckpt.parent/'FINAL_RESULT.json'
    training_result=json.loads(training_result_path.read_text()) if training_result_path.exists() else {}
    report = dict(scope='FROZEN_CHECKPOINT_COUNTERFACTUAL_DIAGNOSTICS_NOT_RETRAINED_ABLATIONS',
        split=args.split, test_read=args.split=='test', samples=total, seed=ck.get('seed'),
        ckpt=str(args.ckpt), checkpoint_sha256=hashlib.sha256(args.ckpt.read_bytes()).hexdigest(),
        task=cfg.task, deploy=cfg.deploy, metrics=report_fn(arrays['deployed'],arrays['labels']),
        data_pipeline=pipeline,
        metrics_history_off=report_fn(arrays['history_off_logits'],arrays['labels']),
        metrics_history_on=report_fn(arrays['history_on_logits'],arrays['labels']),
        groups=groups, contribution_rank_alignment=alignment, gate_counterfactual_counts=gate_accuracy,
        parameters_total=sum(p.numel() for p in model.parameters()),
        parameters_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        clean_ms_per_sample=1000*timers['clean_seconds']/total,
        shapley_ms_per_sample=1000*timers['shapley_seconds']/total,
        latency_scope='batch throughput excluding data adapter, warm-up and host transfers; not end-to-end latency',
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        diagnostic_seconds=time.perf_counter()-began,
        actual_training_seconds=training_result.get('elapsed_sec'),
        epochs_completed=training_result.get('epochs_completed'),
        hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
        torch_version=torch.__version__, external_expert_evidence_present=False,
        transition_definition='Last valid preceding history slot ground-truth class: equal=continuation, unequal=flip. Analysis only; not speaker-aware.',
        regression_loss_definition='squared_error' if cfg.task=='reg' else None,
        missing=['cRBEF expert evidence', 'Full ReCoMER equal-weight/local_current comparisons',
                 'Retrained unified final-model ablations', 'Unseen confirmatory holdout'])
    (args.out/'METRICS.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='DIAGNOSTICS_COMPLETE',out=str(args.out),samples=total,split=args.split)),flush=True)


if __name__ == '__main__':
    main()
