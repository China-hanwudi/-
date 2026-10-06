"""Summarize paired frozen-base runs; incomplete runs are never imputed."""
import argparse
import json
from pathlib import Path

import numpy as np

from gate_confirm_stats import PILOT, SEEDS, stats

VARIANTS = ('constant_task', 'mlp_task', 'evidence_task')
REGRESSION = {'mosei', 'mosei_full', 'mosi', 'chsims'}


def summarize(root, tag, seeds):
    key = 'mae' if tag in REGRESSION else 'weighted_f1'
    records = {}
    missing = []
    for seed in seeds:
        records[seed] = {}
        for variant in VARIANTS:
            path = root/tag/('seed'+str(seed))/variant/'FINAL_RESULT.json'
            if not path.exists():
                missing.append(str(path))
                continue
            r = json.loads(path.read_text(encoding='utf-8'))
            assert r['seed'] == seed and r['variant'] == variant
            assert r['test_read'] is False and r['status'] == 'TRAIN_COMPLETE'
            records[seed][variant] = r
    complete = [s for s in seeds if len(records[s]) == len(VARIANTS)]
    out = dict(metric=key, planned_seeds=list(seeds), complete_seeds=complete,
               missing=missing, all_complete=not missing, test_read=False,
               scope='fixed valid split; seed uncertainty, not population generalization')
    if len(complete) < 2:
        return out
    values = {'uniform': [records[s]['evidence_task']['base_valid'][key] for s in complete]}
    for v in VARIANTS:
        values[v] = [records[s][v]['best_valid'][key] for s in complete]
    for s in complete:
        base = records[s]['evidence_task']['base_valid'][key]
        assert all(abs(records[s][v]['base_valid'][key]-base) < 1e-8 for v in VARIANTS)
    out['means'] = {v: float(np.mean(a)) for v,a in values.items()}
    out['pairs'] = [{ 'seed': s, **{v: a[i] for v,a in values.items()},
                     'evidence_mean_weight': records[s]['evidence_task']['mean_weight']}
                    for i,s in enumerate(complete)]
    out['comparisons'] = {}
    direction = 1 if key == 'mae' else -1
    for reference in ('uniform', 'constant_task', 'mlp_task'):
        gain = direction*(np.array(values[reference])-np.array(values['evidence_task']))
        r = stats(gain)
        nonpilot = [g for s,g in zip(complete,gain) if s not in PILOT]
        if len(nonpilot) >= 2:
            r['nonpilot'] = stats(nonpilot)
        out['comparisons'][reference] = r
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--datasets', required=True)
    p.add_argument('--screen', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    seeds = PILOT if args.screen else SEEDS
    out = dict(status='exploratory 3-seed screen' if args.screen else '10-seed confirmation',
               datasets={tag:summarize(args.root,tag,seeds) for tag in args.datasets.split(',')},
               higher_gain_is_better=True, test_read=False)
    # Adjust only complete datasets, with three pre-specified references each.
    # This does NOT correct for the earlier architecture search or feature search.
    comparisons = [(tag,ref,r) for tag,d in out['datasets'].items() if d['all_complete']
                   for ref,r in d.get('comparisons',{}).items()]
    family = len(comparisons)
    adjusted = 0.
    for i,(tag,ref,r) in enumerate(sorted(comparisons,key=lambda x:x[2]['sign_flip_p_two_sided'])):
        adjusted = max(adjusted,min(1.,(family-i)*r['sign_flip_p_two_sided']))
        r['holm_p_current_complete_family'] = adjusted
    out['holm_family_comparisons'] = family
    out['all_complete'] = all(d['all_complete'] for d in out['datasets'].values())
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(out,indent=2),encoding='utf-8')
    for tag,d in out['datasets'].items():
        print(json.dumps(dict(dataset=tag,n=len(d['complete_seeds']),
                              all_complete=d['all_complete'],means=d.get('means'),
                              comparisons=d.get('comparisons')),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
