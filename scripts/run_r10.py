#!/usr/bin/env python3
"""Sequential desktop R10 run. One heavy process, isolated caches, checked resume."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    """Atomic and durable: a crash leaves either the old or the new file, never zeros."""
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as fh:
        fh.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def fingerprint(paths):
    result = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir() and (path / '_INCOMPLETE').exists():
            raise RuntimeError(f'Incomplete stage output: {path}')
        files = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
        if not files:
            raise RuntimeError(f'Empty stage output: {path}')
        result.extend({'path': str(p), 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in files)
    return result


def code_hash():
    files = sorted((ROOT / 'code/business_entity_resolution/src').rglob('*.py'))
    files += sorted((ROOT / 'scripts/analysis').glob('*.py'))
    files += [Path(__file__), ROOT / 'code/business_entity_resolution/requirements_r10.txt']
    files += [ROOT / 'code/business_entity_resolution/requirements_v2.txt']
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, default=Path('work/r10'))
    p.add_argument('--dataset', type=Path, default=Path('student_resource/dataset'))
    p.add_argument('--output', type=Path, default=Path('output/r10'))
    p.add_argument('--encoder', type=Path, help='Read-only reuse of an R8 encoder trained ONLY on folds 0/1/8/9')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--threads', type=int, default=12)
    p.add_argument('--rounds', type=int, default=1500)
    p.add_argument('--ce-batch', type=int, default=16)
    p.add_argument('--ce-score-batch', type=int, default=128)
    p.add_argument('--ce-epochs', type=int, default=3)
    p.add_argument('--ce-accumulation', type=int, default=4)
    p.add_argument('--ce-checkpointing', action=argparse.BooleanOptionalAction, default=True,
                   help='activation checkpointing (saves VRAM, costs speed; off on 24 GB GPUs)')
    p.add_argument('--ann', choices=['faiss', 'gpu-exact'], default='faiss',
                   help='neighbour search: CPU FAISS IVF (approximate) or exact cosine on the GPU')
    p.add_argument('--shard-pairs', type=int, default=2000000)
    p.add_argument('--r11-features', action='store_true',
                   help='stage-2 --lookalike and --record-competition features')
    p.add_argument('--encode-batch', type=int, default=256)
    p.add_argument('--nprobe', type=int, default=96)
    p.add_argument('--search-k', type=int, default=96)
    p.add_argument('--neural-k', type=int, default=24)
    p.add_argument('--rescue-k', type=int, default=8)
    p.add_argument('--max-hours', type=float, default=24)
    p.add_argument('--reserve-gb', type=float, default=12)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--plan', action='store_true')
    p.add_argument('--preflight', action='store_true')
    return p


def commands(args):
    py = sys.executable
    w, m = args.work, args.work / 'models'
    def module(name, *more):
        return [py, '-u', '-m', 'er_v2.' + name, *map(str, more), '--work', str(w)]
    runtime = ['--device', args.device, '--threads', str(args.threads), '--batch-rows', '500000', '--model-dir', str(m)]
    data = ['--dataset', str(args.dataset)]
    profile = 'enhanced'
    shift = w / 'shift_check.json'
    if shift.exists() and not json.loads(shift.read_text())['allow_enhanced']:
        profile = 'baseline'
    trainopts = ['--rounds', str(args.rounds), '--feature-profile', profile, '--country-thresholds']
    stages = {}
    def add(name, command, *outputs):
        stages[name] = (command, list(outputs))
    add('translit', [py, '-m', 'er_v2.translit', *data, '--out', str(w / 'translit.json')], w / 'translit.json')
    add('prepare', module('prepare', *data, '--workers', args.threads, '--buffer-rows', 100000, '--translit', w / 'translit.json'), w / 'norm')
    for split in ('train', 'test'):
        add('key_' + split, module('run_block', '--split', split, '--phonetic-k', 8, '--candidate-prefix', 'key'),
            w / f'key_{split}.parquet', w / f'key_{split}.json')
    encoder = args.encoder or w / 'neural_e5'
    if args.encoder is None:
        add('encoder', module('neural', 'finetune', *data, '--model-dir', encoder, '--batch', 64, '--epochs', 1), encoder)
    for split in ('train', 'test'):
        add('encode_' + split, module('r10_retrieval', 'encode', '--split', split, '--model-dir', encoder,
            '--batch', args.encode_batch, '--device', args.device), *[w / 'emb' / f'{split}_{side}.npy' for side in ('s1', 'tg')], w / 'emb' / f'{split}.json')
        if args.ann == 'gpu-exact':  # exact top-k on the GPU: ~10 min instead of ~2 h, no approximation
            add('ann_' + split, module('neural', 'block', '--split', split, '--k', args.neural_k), w / f'ncands_{split}.parquet')
        else:
            add('ann_' + split, module('r10_retrieval', 'block', '--split', split, '--k', args.neural_k,
                '--nprobe', args.nprobe, '--search-k', args.search_k, '--threads', args.threads), w / f'ncands_{split}.parquet', w / f'ann_{split}.json')
        add('merge_' + split, module('r10_retrieval', 'merge', '--split', split, '--k', args.neural_k), w / f'cands_{split}.parquet')
        add('features_' + split, module('run_features', '--split', split, *data, '--enhanced', '--workers', args.threads,
            '--shard-pairs', args.shard_pairs, '--overwrite'), w / f'feats_{split}')
    add('shift_check', [py, '-u', str(ROOT / 'scripts/analysis/r7_shift_check.py'), '--work', str(w), '--threads', str(args.threads)], w / 'shift_check.json')
    add('train', module('train', *data, *runtime, *trainopts, '--neural', '--neural-rescue-k', args.rescue_k,
        *(['--lookalike', '--record-competition'] if args.r11_features else []),
        '--max-depth', 8, '--hist-cache-nodes', 1024), w / 'stage2_train.parquet', w / 'eval_preds.parquet',
        *[m / n for n in ('stage1.json', 'stage1_b.json', 'stage2.json', 'metrics.json')])
    add('predict', module('predict', *runtime, '--output', args.output / 'stage2'), w / 'test_preds.parquet')
    for split in ('train', 'test'):
        add('graph_' + split, module('stage3', '--split', split, *data, *runtime, *trainopts,
            '--support-anchors', 1000, '--hist-cache-nodes', 1024, '--output', args.output / 'reference'),
            w / f'stage3_{split}.parquet', w / ('eval_preds_stage3.parquet' if split == 'train' else 'test_preds_stage3.parquet'),
            *([m / 'stage3.json', m / 'stage3_metrics.json'] if split == 'train' else []))
    for split in ('train', 'test'):
        add('tokens_' + split, module('r10_ce', 'tokens', '--split', split),
            *[w / 'ce_tokens' / f'{split}_{side}{suffix}.npy' for side in ('s1', 'tg') for suffix in ('', '_lengths')],
            w / 'ce_tokens' / f'{split}.json')
    ceopts = ['--device', args.device, '--threads', str(min(args.threads, 8)), '--batch', str(args.ce_batch),
              '--score-batch', str(args.ce_score_batch), '--epochs', str(args.ce_epochs),
              '--accumulation', str(args.ce_accumulation),
              *([] if args.ce_checkpointing else ['--no-checkpointing'])]
    add('ce_train', module('r10_ce', 'train', *data, *ceopts), w / 'ce_model')
    for split in ('train', 'test'):
        add('ce_score_' + split, module('r10_ce', 'score', '--split', split, *ceopts), w / f'ce_{split}.parquet')
    finalopts = [*data, '--output', str(args.output), '--device', args.device, '--threads', str(args.threads), '--rounds', str(args.rounds)]
    add('fit', module('r10', 'fit', *finalopts), w / 'r10_models.json', m / 'r10_pair.json', m / 'r10_business.json')
    add('select', module('r10', 'select', *finalopts), w / 'selection.json')
    add('evaluate', module('r10', 'evaluate', *finalopts), w / 'metrics.json')
    add('inference', module('r10', 'inference', *finalopts), w / 'r10_test_predictions.parquet',
        args.output / 'candidate_pairs.tsv', args.output / 'matching_results.tsv')
    add('validate', [py, str(ROOT / 'student_resource/utils/validate_submission.py'), '--matching', str(args.output / 'matching_results.tsv'),
        '--candidate', str(args.output / 'candidate_pairs.tsv'), '--test-dir', str(args.dataset / 'test'), '--check-ids'])
    return stages


def preflight(args):
    import torch
    import faiss
    import transformers
    import sentence_transformers
    import xgboost as xgb
    import numpy as np
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable: install the CUDA torch wheel before running R10')
        props = torch.cuda.get_device_properties(0)
        if props.total_memory < 10 * 1024**3:
            raise RuntimeError('Default profile needs at least 10 GiB VRAM')
        # Fail if XGBoost silently falls back to CPU while PyTorch can use CUDA.
        model = xgb.train({'device': 'cuda', 'tree_method': 'hist'}, xgb.DMatrix(np.eye(4), label=[0, 1, 0, 1]), 1)
        if json.loads(model.save_config())['learner']['generic_param']['device'] == 'cpu':
            raise RuntimeError('XGBoost CUDA support is unavailable')
        print(f'GPU: {props.name}; {props.total_memory / 1024**3:.1f} GiB', flush=True)
    elif args.encoder is None:
        raise RuntimeError('Fresh encoder fine-tuning uses CUDA; CPU runs need --encoder (CPU is for smoke tests)')
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'sentence-transformers', 'faiss-cpu', 'xgboost', 'polars', 'numpy', 'pyarrow')}
    if args.encoder:
        meta = json.loads((args.encoder / 'finetune.json').read_text())
        if sorted(meta['encoder_folds']) != [0, 1, 8, 9]:
            raise ValueError('Encoder provenance must specify folds 0/1/8/9 only')
    for split in ('train', 'test'):
        for side in (1, 2, 3):
            if not (args.dataset / split / f'{split}_source{side}.tsv').is_file():
                raise FileNotFoundError(f'Missing {split} source{side}')
    if not (args.dataset / 'train/train_ground_truth.tsv').exists():
        raise FileNotFoundError('Training ground truth missing')
    if not (ROOT / 'student_resource/utils/validate_submission.py').exists():
        raise FileNotFoundError('Copy the organizer validator to student_resource/utils/validate_submission.py')
    print(json.dumps({'versions': versions, 'free_gb': shutil.disk_usage(args.work).free / 1024**3}, indent=2))
    return versions


def stop_child(child):
    child.terminate()
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=10)


def main():
    p = parser()
    args = p.parse_args()
    if min(args.threads, args.rounds, args.ce_batch, args.ce_score_batch, args.ce_epochs, args.encode_batch,
           args.nprobe, args.neural_k, args.search_k) < 1 or not 0 <= args.rescue_k <= args.neural_k or args.max_hours <= 0 or args.reserve_gb < 1:
        p.error('Invalid counts, rescue size, storage reserve or deadline')
    for name in ('work', 'dataset', 'output', 'encoder'):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    if args.work == args.dataset or args.output == args.work or args.dataset.is_relative_to(args.work):
        p.error('Work, dataset and output must be separate paths')
    if args.plan:
        for stage, (command, _) in commands(args).items():
            print(stage, subprocess.list2cmdline(command))
        return
    args.work.mkdir(parents=True, exist_ok=True)
    versions = preflight(args)
    if args.preflight:
        return
    lock = args.work / 'run.lock'
    # Kernel file lock: automatically released after crashes, works on Windows/WSL/Linux/macOS.
    import filelock
    with filelock.FileLock(str(lock), timeout=0):
        (args.work / 'models').mkdir(exist_ok=True)
        (args.work / 'logs').mkdir(exist_ok=True)
        manifest = args.work / 'run.json'
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ('resume', 'max_hours', 'plan', 'preflight')}
        identity = {'code': code_hash(), 'settings': config, 'versions': versions,
            'data': fingerprint([args.dataset / 'train', args.dataset / 'test']),
            'encoder': fingerprint([args.encoder]) if args.encoder else None,
            'validator': fingerprint([ROOT / 'student_resource/utils/validate_submission.py'])}
        if args.resume:
            state = json.loads(manifest.read_text())
            if state['identity'] != identity:
                raise RuntimeError('Data/code/settings/dependencies changed: use a fresh work directory')
        else:
            if manifest.exists() or any((args.work / 'norm').glob('*.parquet')):
                raise RuntimeError('Existing work directory: use --resume or a fresh --work')
            state = {'identity': identity, 'started': now(), 'completed': {}}
        state.update(status='running', pid=os.getpid())
        env = {**os.environ, 'PYTHONPATH': str(ROOT / 'code/business_entity_resolution/src') + os.pathsep + os.environ.get('PYTHONPATH', ''),
            'POLARS_MAX_THREADS': str(args.threads), 'OMP_NUM_THREADS': str(args.threads), 'OPENBLAS_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUTF8': '1', 'R10_NEURAL_DEVICE': args.device}
        deadline = time.monotonic() + args.max_hours * 3600
        try:
            for stage in commands(args):
                command, outputs = commands(args)[stage]
                previous = state['completed'].get(stage)
                if previous:
                    if previous['command'] != command or previous['outputs'] != fingerprint(outputs):
                        raise RuntimeError(f'Completed stage changed: {stage}')
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError('R10 wall-time limit reached; --resume restarts the interrupted stage')
                if shutil.disk_usage(args.work).free < args.reserve_gb * 1024**3:
                    raise RuntimeError('Free disk below reserve; add space before resuming')
                logpath = args.work / 'logs' / f'{stage}.log'
                state.update(stage=stage, log=str(logpath), stage_started=now())
                save(manifest, state)
                print(f'{now()} {stage} -> {logpath}', flush=True)
                started = time.monotonic()
                with logpath.open('w', encoding='utf-8') as log:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        while child.poll() is None:
                            time.sleep(2)
                            if time.monotonic() >= deadline:
                                raise TimeoutError(f'{stage} reached R10 wall-time limit')
                            state['stage_seconds'] = round(time.monotonic() - started, 1)
                            save(manifest, state)
                    except BaseException:
                        stop_child(child)
                        raise
                if child.returncode:
                    raise RuntimeError(f'{stage} failed ({child.returncode}); inspect {logpath}')
                state['completed'][stage] = {'command': command, 'outputs': fingerprint(outputs),
                    'seconds': round(time.monotonic() - started, 1)}
                save(manifest, state)
            metrics = json.loads((args.work / 'metrics.json').read_text())
            result = {'status': 'complete', 'official_validation': 'PASS', 'metrics': metrics,
                'output_sha256': {name: file_hash(args.output / name) for name in ('matching_results.tsv', 'candidate_pairs.tsv')},
                'stage_seconds': {k: v['seconds'] for k, v in state['completed'].items()}}
            save(args.work / 'result.json', result)
            value = metrics['local_fold4']['macro_f05']
            (args.work / 'result.md').write_text(f'R10 complete. Official TSV validation: PASS.\n\nSelected: {metrics["selected"]}. Local fold-4 macro F0.5: {value:.6f}.\n\n97.5 local target met: {value >= .975}. Amazon leaderboard score: not measured.\n', encoding='utf-8')
            state.update(status='complete', finished=now())
            save(manifest, state)
            print(f'Complete: {args.work / "result.md"}')
        except BaseException as exc:
            state.update(status='failed', error=str(exc), finished=now())
            save(manifest, state)
            save(args.work / 'result.json', {'status': 'failed', 'error': str(exc), 'finished': now()})
            raise


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()

if __name__ == '__main__':
    main()
