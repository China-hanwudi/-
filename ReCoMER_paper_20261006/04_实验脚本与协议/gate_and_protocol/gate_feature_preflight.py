"""Read train/valid ONLY, audit available representation packs before training."""
import json
from pathlib import Path

import torch

from n6.data import open_split, split_sha256

ROOT = Path('/data/emo/肖田泽科研/数据')
REFERENCE = {
    'mosei': ROOT/'MOSEI/packed',
    'meld': ROOT/'MELD/packed',
    'm3ed': ROOT/'M3ED/packed_audio_e2_zh_hubert_large',
}
CANDIDATES = {
    'mosei_full': ('mosei', 'MOSEI_full/packed', 'reg'),
    'mosei_qwen': ('mosei', 'MOSEI_textQwen_wavlm/packed', 'reg'),
    'meld_roberta': ('meld', 'MELD_robertaFT/packed', 'cls'),
    'm3ed_roberta': ('m3ed', 'M3ED_textRobertaFT/packed', 'cls'),
    'm3ed_ftT': ('m3ed', 'M3ED/packed_official_ftT', 'cls'),
    'mosi': (None, 'CMU-MOSI/packed', 'reg'),
    'chsims': (None, 'CH-SIMS_v2/full_packed', 'reg'),
}


def ids(ds):
    return [str(x) for x in ds.raw.get('ids', [])]


def audit_one(tag, parent, relative, task):
    path = ROOT/relative
    result = {'tag': tag, 'path': str(path), 'task': task, 'splits': {},
              'test_read': False, 'provenance': {}}
    for file in sorted(path.glob('*.json')):
        if 'sealed' in file.name.lower():
            continue
        result['provenance'][file.name] = json.loads(file.read_text())
    split_ids = {}
    for split in ('train.pt', 'valid.pt'):
        ds = open_split(path, split, task)
        b, y = ds.batch(range(min(16, ds.n)))
        split_ids[split] = ids(ds)
        row = {'n': ds.n, 'dims': ds.dims, 'history_k': ds.k,
               'keys': list(ds.raw), 'sha256': split_sha256(ds),
               'missing_counts': (ds.modality_mask <= 0).sum(0).tolist(),
               'finite_features': all(torch.isfinite(ds.raw[m]).all().item() for m in ('T','A','V')),
               'label_min': float(ds.raw['label'].min()),
               'label_max': float(ds.raw['label'].max()),
               'batch_shapes': {k:list(v.shape) for k,v in b.items()}}
        if parent:
            ref = open_split(REFERENCE[parent], split, task)
            row.update(same_ordered_ids_as_reference=(ids(ds)==ids(ref) and len(ids(ds))==ds.n),
                       same_id_set_as_reference=(set(ids(ds))==set(ids(ref)) and len(ids(ds))==ds.n),
                       same_ordered_labels_as_reference=torch.equal(ds.raw['label'],ref.raw['label']),
                       same_mask_as_reference=torch.equal(ds.modality_mask,ref.modality_mask),
                       same_history_as_reference=torch.equal(ds.raw.get('history_index',torch.empty(0)),ref.raw.get('history_index',torch.empty(0))))
        result['splits'][split] = row
    result['train_valid_id_overlap'] = len(set(split_ids['train.pt']) & set(split_ids['valid.pt']))
    result['no_duplicate_ids'] = all(len(v)==len(set(v)) for v in split_ids.values())
    return result


def main():
    torch.set_num_threads(2)
    rows = []
    for tag,(parent,relative,task) in CANDIDATES.items():
        try:
            row = audit_one(tag,parent,relative,task)
        except Exception as exc:
            row = {'tag':tag,'error':str(exc),'test_read':False}
        rows.append(row)
        print(json.dumps(row),flush=True)
    out = Path('/data/emo/肖田泽科研/模型/model6_innovation_validation_20260930/experiments/gate_feature_screen_20261002')
    out.mkdir(parents=True,exist_ok=True)
    target = out/'feature_preflight.json'
    if target.exists():
        raise SystemExit('refusing overwrite preflight')
    target.write_text(json.dumps({'packs':rows,'test_read':False},indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
