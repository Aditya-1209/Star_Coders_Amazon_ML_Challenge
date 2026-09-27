#!/usr/bin/env python3
"""Bounded R12-France follow-up using immutable, completed R12 artifacts."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from run_r10 import ROOT, now, save, fingerprint, code_hash, file_hash, stop_child


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-work', type=Path, default=Path('work/r12'))
    p.add_argument('--work', type=Path, default=Path('work/r12_france'))
    p.add_argument('--output', type=Path, default=Path('output/r12_france'))
    p.add_argument('--dataset', type=Path, default=Path('student_resource/dataset'))
    p.add_argument('--validator', type=Path, default=Path('student_resource/utils/validate_submission.py'))
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--threads', type=int, default=30)
    p.add_argument('--rounds', type=int, default=900)
    p.add_argument('--max-hours', type=float, default=3.5)
    p.add_argument('--reserve-gb', type=float, default=20)
    p.add_argument('--plan', action='store_true')
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--resume', action='store_true')
    return p


def commands(args):
    shared = ['--base-work', args.base_work, '--work', args.work, '--output', args.output,
              '--dataset', args.dataset, '--device', args.device, '--threads', args.threads,
              '--rounds', args.rounds]
    stages = {}
    for name, stage, extra, outputs in [
        ('features_train', 'prepare', ['--split', 'train'], ['features_train.parquet']),
        ('features_test', 'prepare', ['--split', 'test'], ['features_test.parquet']),
        ('fit', 'fit', [], ['models.json', 'models/balanced.json', 'models/transfer.json', 'models/domain.json']),
        ('select', 'select', [], ['selection.json']),
        ('features_holdout', 'prepare', ['--split', 'holdout'], ['features_holdout.parquet']),
        ('evaluate', 'evaluate', [], ['metrics.json']),
        ('inference', 'inference', [], ['france_predictions.parquet', 'changes.json']),
    ]:
        stages[name] = ([sys.executable, '-u', '-m', 'er_v2.r12_france', stage, *map(str, shared), *extra],
                        [args.work / path for path in outputs])
    stages['inference'][1].extend(args.output / f for f in ('matching_results.tsv', 'candidate_pairs.tsv'))
    stages['validate'] = ([sys.executable, str(args.validator), '--matching', str(args.output / 'matching_results.tsv'),
        '--candidate', str(args.output / 'candidate_pairs.tsv'), '--test-dir', str(args.dataset / 'test'), '--check-ids'], [])
    return stages


def parent_paths(args):
    names = ['run.json', 'result.json', 'selection.json', 'metrics.json', 'r10_models.json',
             'shift_check.json', 'ce_model/training.json', 'neural_e5/finetune.json',
             'models/stage3_metrics.json', 'models/r10_pair.json', 'models/r10_business.json',
             'eval_preds_stage3.parquet', 'r10_test_predictions.parquet']
    names += [f'{prefix}_{split}.parquet' for prefix in ('stage3', 'ce') for split in ('train', 'test')]
    names += [f'norm/{split}_source{i}.parquet' for split in ('train', 'test') for i in (1, 2, 3)]
    return [args.base_work / name for name in names]


def verify_parent_files(args, state):
    # Verify that cache/model files still match the completed producer manifest,
    # not just that their paths exist. Never bypass the old runner's code check
    # by editing its manifest: this separate runner reads its artifacts only.
    recorded = {item['path']: item for stage in state['completed'].values()
                for item in stage['outputs']}
    for item in fingerprint(parent_paths(args)):
        path = Path(item['path'])
        if path.name in ('run.json', 'result.json'):
            continue
        if recorded.get(item['path']) != item:
            raise ValueError(f'Parent artifact no longer matches its completed stage: {path}')


def preflight(args):
    import polars as pl
    import xgboost as xgb
    fingerprint(parent_paths(args))
    state = json.loads((args.base_work / 'run.json').read_text())
    result = json.loads((args.base_work / 'result.json').read_text())
    if state.get('status') != 'complete' or result.get('official_validation') != 'PASS':
        raise ValueError('R12-France requires a completed, officially validated R12 run')
    verify_parent_files(args, state)
    selection = json.loads((args.base_work / 'selection.json').read_text())
    if selection.get('version') != 'r10-ce-ann-1' or selection.get('selected') not in ('r10', 'reference'):
        raise ValueError('Expected the reviewed R12 fusion artifacts; other versions need an explicit migration')
    ce = json.loads((args.base_work / 'ce_model/training.json').read_text())
    encoder = json.loads((args.base_work / 'neural_e5/finetune.json').read_text())
    if sorted(ce['train_folds']) != [0, 1, 8] or ce['early_stopping_folds'] != [9] or sorted(encoder['encoder_folds']) != [0, 1, 8, 9]:
        raise ValueError('Parent neural fold provenance is incompatible')
    current_data = fingerprint([args.dataset / 'train', args.dataset / 'test'])
    if current_data != state['identity']['data']:
        raise ValueError('Dataset differs from the completed R12 run; use the same original data path and files')
    if fingerprint([args.validator]) != state['identity']['validator']:
        raise ValueError('Validator differs from the completed R12 run')
    # Check required schema without loading the feature tables. Pair-key
    # uniqueness is checked when evidence is constructed.
    for split in ('train', 'test'):
        schema = pl.scan_parquet(args.base_work / f'stage3_{split}.parquet').collect_schema()
        if not {'sidx', 'tidx', 'core_tset', 'addr_tset', 'direct'} <= set(schema.names()):
            raise ValueError('Parent stage-3 schema is incompatible')
    if args.device == 'cuda':
        import numpy as np
        import torch
        if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 20 * 1024**3:
            raise ValueError('The VM profile requires an L4-class GPU with 24 GB VRAM')
        booster = xgb.train({'device': 'cuda', 'tree_method': 'hist'}, xgb.DMatrix(np.eye(4), label=[0, 1, 0, 1]), 1)
        if json.loads(booster.save_config())['learner']['generic_param']['device'] == 'cpu':
            raise ValueError('XGBoost CUDA is unavailable')
    if shutil.disk_usage(args.work).free < args.reserve_gb * 1024**3:
        raise ValueError('Free disk is below the reserve; retain R12 and add space before running')
    versions = {name: importlib.metadata.version(name) for name in ('polars', 'numpy', 'pyarrow', 'xgboost', 'torch')}
    if any(state['identity']['versions'].get(name) != version for name, version in versions.items()):
        raise ValueError('Dependencies changed since R12; restore its environment before reusing fold/model artifacts')
    print(json.dumps({'parent': str(args.base_work), 'parent_local_score': result['metrics']['local_fold4'],
                      'versions': versions, 'free_gib': shutil.disk_usage(args.work).free / 1024**3}), flush=True)
    return versions


def main():
    p = parser()
    args = p.parse_args()
    if min(args.threads, args.rounds) < 1 or not 0 < args.max_hours <= 4 or not 1 <= args.reserve_gb <= 1000:
        p.error('Invalid counts, deadline or disk reserve')
    for name in ('work', 'base_work', 'output', 'dataset', 'validator'):
        setattr(args, name, getattr(args, name).resolve())
    paths = [args.base_work, args.work, args.output, args.dataset]
    if any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
           for i, a in enumerate(paths) for b in paths[i + 1:]):
        p.error('Base, work, output and dataset must be separate non-nested directories')
    plan = commands(args)
    if args.plan:
        for name, (command, _) in plan.items():
            print(name, subprocess.list2cmdline(command))
        return
    args.work.mkdir(parents=True, exist_ok=True)
    versions = preflight(args)
    if args.preflight:
        return
    import filelock
    with filelock.FileLock(str(args.base_work / 'run.lock'), timeout=0), \
         filelock.FileLock(str(args.work / 'run.lock'), timeout=0):
        (args.work / 'logs').mkdir(exist_ok=True)
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                  if k not in ('resume', 'max_hours', 'plan', 'preflight')}
        identity = {'code': hashlib.sha256((code_hash() + file_hash(Path(__file__))).encode()).hexdigest(),
                    'config': config, 'versions': versions, 'parent': fingerprint(parent_paths(args)),
                    'data': fingerprint([args.dataset / 'train', args.dataset / 'test'])}
        manifest = args.work / 'run.json'
        if args.resume:
            state = json.loads(manifest.read_text())
            if state['identity'] != identity:
                raise ValueError('R12-France code/settings/data/parent changed; use a fresh work directory')
        else:
            if manifest.exists():
                raise ValueError('Existing R12-France run; use --resume with identical settings')
            state = {'identity': identity, 'started': now(), 'completed': {}}
        state.update(status='running', pid=os.getpid())
        save(manifest, state)
        env = {**os.environ, 'PYTHONPATH': str(ROOT / 'code/business_entity_resolution/src'),
               'POLARS_MAX_THREADS': str(args.threads), 'OMP_NUM_THREADS': str(args.threads),
               'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'PYTHONUTF8': '1'}
        deadline = time.monotonic() + args.max_hours * 3600
        def interrupted(signum, frame):
            raise KeyboardInterrupt('R12-France interrupted')
        old = signal.signal(signal.SIGTERM, interrupted)
        try:
            for name, (command, outputs) in plan.items():
                previous = state['completed'].get(name)
                if previous:
                    if previous['command'] != command or previous['outputs'] != fingerprint(outputs):
                        raise ValueError(f'Completed R12-France stage changed: {name}')
                    continue
                if time.monotonic() >= deadline or shutil.disk_usage(args.work).free < args.reserve_gb * 1024**3:
                    raise RuntimeError('R12-France deadline or disk reserve reached')
                state.update(stage=name, stage_started=now())
                save(manifest, state)
                logpath = args.work / 'logs' / f'{name}.log'
                print(f'{now()} {name} -> {logpath}', flush=True)
                started = time.monotonic()
                with logpath.open('w', encoding='utf-8') as log:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=os.name == 'posix')
                    try:
                        while child.poll() is None:
                            time.sleep(2)
                            if time.monotonic() >= deadline or shutil.disk_usage(args.work).free < args.reserve_gb * 1024**3:
                                raise RuntimeError(f'{name}: R12-France deadline or disk reserve reached')
                    except BaseException:
                        stop_child(child)
                        raise
                if child.returncode:
                    raise RuntimeError(f'{name} failed ({child.returncode}); inspect {logpath}')
                state['completed'][name] = {'command': command, 'outputs': fingerprint(outputs),
                                           'seconds': round(time.monotonic() - started, 1)}
                save(manifest, state)
            if fingerprint(parent_paths(args)) != identity['parent']:
                raise RuntimeError('Parent R12 artifacts changed during the run')
            metrics = json.loads((args.work / 'metrics.json').read_text())
            result = {'status': 'complete', 'official_validation': 'PASS', 'metrics': metrics,
                      'changes': json.loads((args.work / 'changes.json').read_text()),
                      'adaptation': json.loads((args.work / 'models.json').read_text())['domain'],
                      'output_sha256': {name: file_hash(args.output / name) for name in ('matching_results.tsv', 'candidate_pairs.tsv')},
                      'stage_seconds': {k: v['seconds'] for k, v in state['completed'].items()}}
            save(args.work / 'result.json', result)
            (args.work / 'result.md').write_text(f'R12-France complete. Official validation: PASS.\n\n'
                f'Selected: {metrics["selected"]}. Fold-4 F0.5: {metrics["local_fold4"]["macro_f05"]:.6f}.\n\n'
                'Only France may change. France/leaderboard accuracy is unmeasured; local score evaluates unchanged R12.\n', encoding='utf-8')
            state.update(status='complete', finished=now())
            print(f'Complete: {args.work / "result.md"}', flush=True)
        except BaseException as exc:
            state.update(status='failed', error=str(exc), finished=now())
            save(args.work / 'result.json', {'status': 'failed', 'error': str(exc)})
            raise
        finally:
            save(manifest, state)
            signal.signal(signal.SIGTERM, old)


if __name__ == '__main__':
    main()
