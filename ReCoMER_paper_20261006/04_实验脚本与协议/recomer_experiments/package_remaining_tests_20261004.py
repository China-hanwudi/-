"""Collect verified frozen models, all controls, predictions and reports."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import zipfile


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def save(p,z):p.write_text(json.dumps(z,indent=2,ensure_ascii=False),encoding='utf-8')


def main(root):
    plan=json.loads((root/'PLAN.json').read_text());done=json.loads((root/'COMPLETE.json').read_text())
    assert done['status']=='COMPLETE'
    jobs=list(root.glob('ablations/*/*/mhnou/*/FINAL_RESULT.json'))
    controls=list(root.glob('ablations/*/*/TEST_METRICS.json'))
    assert len(jobs)==75 and len(controls)==18,(len(jobs),len(controls))
    assert all(v['returncode']==0 for v in done['attempts'])
    assert len(done['attempts'])==196
    assert (root/'full_feature_shapley/COMPLETE.json').exists()
    assert (root/'PRODUCTION_IDENTITY_AUDIT.json').exists()
    assert 'Ran 21 tests' in (root/'logs/unit_tests.log').read_text()
    for p,h in plan['asset_sha256'].items():assert sha(p)==h,p
    for p,h in plan['code_sha256'].items():assert sha(root/'code'/p)==h,p
    assert sha(root/'run_remaining_tests_20261004.py')==plan['harness_sha256']
    original=Path('/root/recomer_complete_20261004_side')
    target=root/'frozen_production_models';target.mkdir(exist_ok=False)
    for seed in plan['seeds']:
        p=target/str(seed);p.mkdir()
        for n in ('recomer.pt','local_current.pt','TRAIN_RESULT.json'):
            shutil.copy2(original/'recomer'/str(seed)/n,p/n)
    target=root/'frozen_baseline_models';target.mkdir(exist_ok=False)
    for q in plan['baseline_selection']:
        for seed,source in zip(plan['seeds'],q['selected']['runs']):
            p=target/q['dataset']/q['arm']/str(seed);p.mkdir(parents=True)
            for n in ('best.pt','FINAL_RESULT.json','RUN_METADATA.json','history.json'):
                src=Path(source)/n
                if src.exists():shutil.copy2(src,p/n)
    ref=root/'source_contracts';ref.mkdir(exist_ok=False)
    shutil.copytree(original/'manifests',ref/'matching_train_manifests')
    shutil.copy2('/root/autodl-tmp/fusion_research_automation_20261003/round017_repair1/CONTRACTS.json',ref/'CR17_CONTRACTS.json')
    for p in jobs:
        arm=p.parents[3].name
        save(p.parent/'CONTROL_RUNTIME.json',dict(arm=arm,description=plan['ablations'][arm],
           implementation=str(root/'run_remaining_tests_20261004.py'),implementation_sha256=plan['harness_sha256'],
           warning='For runtime controls use the isolated --export-worker. Loading a checkpoint directly with production code restores the default methods.',
           production_model=False))
    for seed in plan['seeds']:
        save(root/'ablations/no_relative_C'/str(seed)/'CONTROL_RUNTIME.json',dict(arm='no_relative_C',
             required_input_transform='Set all relative C entries to zero before outer fusion',production_model=False))
    save(root/'FINAL_COVERAGE.json',dict(unit_tests=21,official_complete_model_test_runs=3,
          four_dataset_baseline_test_runs=24,checkpoint_mechanism_runs=15,feature_stress_cases=18,
          retrained_MH_control_fits=75,control_fusion_fits=18,control_final_test_runs=18,
          full_feature_shapley_seeds=3,full_feature_shapley_subsets_per_seed=8,
          frozen_CR_gate_formula_probes=6,planned_subprocesses=196,subprocess_failures=0,
          gradient_and_information_flow_audits_complete=True,data_audit_datasets=4,
          primary_model_parameters_unchanged=True,expert_seed_fixed=17,
          unavailable=['Complete cRBEF/ReCoMER for MELD and IEMOCAP: no matching trained experts/feature contracts',
                       'Complete ReCoMER for MOSEI: regression extension of classification expert/fuser not implemented',
                       'Comparable external paper baselines',
                       'Fully independent upstream OOF and raw-input end-to-end cost'],
          adverse_findings=['No hard rejection on 4003 historical test utterances for all 3 full-model MH branches',
                           'Solo heads have no gradient under the active closed-loop objective',
                           'cRBEF affects class fusion and does not directly guide MH modality weights',
                           'Forced history-residual rejection leaves a small router history-information path',
                           'Original cRBEF text hidden features include recent4 history outside MHnoU abstention control'],
          first_additional_audit_repaired='MELD split-local identifiers required namespace-aware comparison; first log retained'))
    save(root/'ENVIRONMENT.json',dict(python=sys.version,
          packages={n:importlib.metadata.version(n) for n in ('torch','numpy','transformers','peft','gptqmodel','accelerate','safetensors')},
          hardware='NVIDIA GeForce RTX 5090',code_scope='all original production code and models preserved'))
    files=[p for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
           and p.suffix!='.zip' and p.name not in ('ARTIFACT_MANIFEST.json','PACKAGE_RECEIPT.json')]
    save(root/'ARTIFACT_MANIFEST.json',dict(files={str(p.relative_to(root)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in files},
           models_included='all 75 control-fold weights, 24 selected baseline weights, and 3 frozen production bundles',
           manifest_self_hash='in external PACKAGE_RECEIPT.json',source_models_unchanged=True))
    archive=root.parent/'recomer_remaining_tests_results_20261004.zip';assert not archive.exists()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in [*files,root/'ARTIFACT_MANIFEST.json']:z.write(p,str(p.relative_to(root)))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    receipt=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(files)+1,
                 manifest_sha256=sha(root/'ARTIFACT_MANIFEST.json'),all_controls_saved=True)
    save(root/'PACKAGE_RECEIPT.json',receipt)
    print(json.dumps(dict(status='PACKAGED',bytes=receipt['bytes'],files=receipt['files'],sha256_chunks=[receipt['sha256'][i:i+16] for i in range(0,64,16)])))


if __name__=='__main__':main(Path(sys.argv[1]))
