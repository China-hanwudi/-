"""One-shot GPU queue using only the supplied n6 code, not an external expert.

This is CORE_ONLY. The folder has no cRBEF source-expert/prediction assets.
No external-package weights or predictions are substituted.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--wait-controller', type=int, default=186484)
    args = ap.parse_args()
    root = args.root.resolve()
    code = root / 'code'
    logs = root / 'logs_core'
    logs.mkdir(exist_ok=True)
    own_lock = (root / 'CORE_QUEUE.lock').open('a+')
    fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = root / 'CORE_STATUS.json'
    data_root = Path('/root/autodl-tmp/data')
    datasets = [('M3ED_textQwen', 'cls'), ('MELD', 'cls'),
                ('IEMOCAP', 'cls'), ('MOSEI_full', 'reg')]
    router = ['--utility', 'shapley', '--gate-architecture', 'evidence',
              '--gate-detach-inputs', '--detach-utility-path', '--bounded-w',
              '--bounded-lambda', '0.3', '--history-abstain-variant', 'utility_softmax']
    arms = {
        'no_history': ['--utility', 'uniform', '--deploy', 'closed_loop', '--no-history'],
        'history_uncontrolled': ['--utility', 'uniform', '--deploy', 'closed_loop'],
        'history_admission': ['--utility', 'uniform', '--deploy', 'closed_loop',
                              '--history-abstain-variant', 'utility_softmax'],
        'full_closed_loop': router + ['--deploy', 'closed_loop'],
        'evidence_solo': router + ['--deploy', 'solo_weighted'],
    }
    hashes = {str(p.relative_to(code)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(code.rglob('*.py'))}
    plan = dict(scope='CORE_ONLY_NO_CRBEF_EXPERT', code_sha256=hashes,
                datasets=datasets, arms=arms, seeds=[7, 13, 17], epochs=20,
                batch_size=256, swa_window=3, patience=6,
                train_runs=60, final_test_runs=60,
                selection='valid only; all training must succeed before any final test',
                test_scope='New checkpoint evaluations on previously evaluated packs, not a never-seen test benchmark',
                external_expert_used=False, optional_mpath_enabled=False)
    if (root / 'CORE_PLAN.json').exists():
        raise RuntimeError('This one-shot queue already has a plan; inspect prior outputs before rerunning')
    save(root / 'CORE_PLAN.json', plan)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    started = time.time()
    def state(phase, **details):
        record = dict(phase=phase, pid=os.getpid(), scope=plan['scope'],
                      seconds=round(time.time()-started), **details)
        save(status, record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    def run(cmd, logfile, timeout=7200):
        with logfile.open('w', encoding='utf-8') as f:
            subprocess.run(cmd, cwd=code, env=env, stdout=f,
                           stderr=subprocess.STDOUT, check=True, timeout=timeout)
    handles = []
    try:
        for dataset, _ in datasets:
            for split in ['train.pt', 'valid.pt', 'test.pt']:
                if not (data_root / dataset / 'packed' / split).is_file():
                    raise RuntimeError(f'Missing input file {dataset}/{split}')
        run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
            logs / 'unit_tests.log', timeout=180)
        state('WAITING_FOR_GPU', wait_controller=args.wait_controller,
              test_read=False, training_started=False)
        while True:
            if time.time()-started > 86400:
                raise TimeoutError('GPU wait exceeded 24 hours; no resource preemption')
            proc = Path(f'/proc/{args.wait_controller}/cmdline')
            if proc.exists() and b'round035_upstream_rebuild/run_pipeline.py' in proc.read_bytes():
                time.sleep(30)
                continue
            try:
                for name in ['GPU_RESEARCH.lock', 'FUSION_RESEARCH.lock']:
                    p = Path('/root/autodl-tmp/fusion_research_automation_20261003') / name
                    h = p.open('a+')
                    handles.append(h)
                    fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
                gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                               '--format=csv,noheader,nounits'], text=True).strip()
                if gpu:
                    raise BlockingIOError('Existing GPU process')
                break
            except BlockingIOError:
                for h in handles:
                    h.close()
                handles.clear()
                time.sleep(30)
        if shutil.disk_usage(root).free < 2 * 1024**3:
            raise RuntimeError('Less than 2 GiB disk free; queue refuses to start')
        current = {str(p.relative_to(code)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(code.rglob('*.py'))}
        if current != hashes:
            raise RuntimeError('Source changed while waiting; queue must be re-audited')
        completed = []
        for dataset, task in datasets:
            for arm, extra in arms.items():
                for seed in plan['seeds']:
                    if shutil.disk_usage(root).free < 1024**3:
                        raise RuntimeError('Less than 1 GiB disk free during training')
                    name = f'{dataset}__{arm}__s{seed}'
                    out = root / 'runs_core' / name
                    if out.exists():
                        raise RuntimeError(f'Run output already exists: {out}')
                    state('TRAINING', run=name, completed_train=len(completed), test_read=False)
                    command = [sys.executable, '-u', '-m', 'n6.train',
                               '--data', str(data_root/dataset/'packed'), '--out', str(out),
                               '--task', task, '--seed', str(seed), '--epochs', '20',
                               '--batch-size', '256', '--swa-window', '3', '--patience', '6',
                               '--device', 'cuda', '--tag', 'supplied_folder_core_20261004',
                               '--dump-valid-logits'] + extra
                    run(command, logs / f'{name}.train.log')
                    result = json.loads((out/'FINAL_RESULT.json').read_text())
                    if result.get('test_read') is not False or not (out/'best.pt').is_file():
                        raise RuntimeError(f'Training output failed admission: {name}')
                    completed.append((dataset, name, out))
        state('ALL_TRAINING_COMPLETE', completed_train=len(completed), test_read=False)
        for i, (dataset, name, out) in enumerate(completed):
            state('FINAL_TEST', run=name, completed_train=60, completed_test=i, test_read=True)
            run([sys.executable, '-u', '-m', 'n6.evaluate',
                 '--data', str(data_root/dataset/'packed'), '--ckpt', str(out/'best.pt'),
                 '--split', 'test', '--device', 'cuda', '--batch-size', '256',
                 '--out', str(out/'TEST_RESULT.json')], logs/f'{name}.test.log')
            report = json.loads((out/'TEST_RESULT.json').read_text())
            if report.get('test_read') is not True:
                raise RuntimeError(f'Final evaluation not marked test: {name}')
        state('COMPLETE', completed_train=60, completed_test=60, test_read=True,
              external_fusion_complete=False)
    except BaseException as exc:
        state('FAILED', error=repr(exc))
        raise
    finally:
        for h in handles:
            h.close()
        own_lock.close()


if __name__ == '__main__':
    main()
