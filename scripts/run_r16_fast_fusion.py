#!/usr/bin/env python3
"""Bounded R16 CE fusion from completed artifacts; no upstream training or installs."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone, timedelta
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from run_r10 import ROOT, now, save, fingerprint, file_hash, code_hash, stop_child

VERSION = 'r16-fast-ce-fusion-1'
COMPONENTS = ('graph', 'ce_a', 'ce_a_swapped', 'ce_b')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base-work', 'r12-export', 'r14-provenance', 'dataset', 'validator'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--work', type=Path, default=Path('work/r16_fast_ce_fusion'))
    p.add_argument('--output', type=Path, default=Path('output/r16_fast_ce_fusion'))
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--threads', type=int, default=24)
    p.add_argument('--rounds', type=int, default=400)
    p.add_argument('--max-minutes', type=float, default=45)
    p.add_argument('--shutdown-at', help='Required UTC/offset ISO time of the EXISTING VM shutdown')
    p.add_argument('--download-buffer-minutes', type=float, default=10)
    p.add_argument('--reserve-gb', type=float, default=20)
    for flag in ('resume', 'plan', 'preflight'):
        p.add_argument('--' + flag, action='store_true')
    return p


def remaining_seconds(args, current=None):
    if not args.shutdown_at:
        raise ValueError('--shutdown-at is required; this runner never changes the VM shutdown timer')
    end = datetime.fromisoformat(args.shutdown_at.replace('Z', '+00:00'))
    if end.tzinfo is None:
        raise ValueError('--shutdown-at must include a timezone (Z for UTC)')
    seconds = (end - timedelta(minutes=args.download_buffer_minutes) - (current or datetime.now(timezone.utc))).total_seconds()
    if seconds <= 0:
        raise TimeoutError('VM shutdown/download deadline has already passed')
    return min(seconds, args.max_minutes * 60)


def safe_paths(args, read_paths=()):
    protected = [args.base_work, args.r12_export, args.r14_provenance, args.dataset, args.validator, *read_paths]
    for write in (args.work, args.output):
        for read in protected:
            if write == read or write.is_relative_to(read) or read.is_relative_to(write):
                raise ValueError(f'Output/work must be separate from original artifacts: {write} / {read}')
    if args.work == args.output or args.work.is_relative_to(args.output) or args.output.is_relative_to(args.work):
        raise ValueError('Work and output must be separate, non-nested directories')


def donor_files(path):
    meta = json.loads(path.read_text())
    if meta.get('score_stage') != 'pre_fusion':
        raise ValueError('Need PRE-fusion CE predictions. Final cross-fitted R14 scores do not preserve the 3B gate.')
    if not meta.get('model_id') or not meta.get('producer_note'):
        raise ValueError('Record the real R14 model/run identity and producer provenance note')
    if set(meta.get('components', {})) != set(COMPONENTS):
        raise ValueError('Required components: graph, ce_a, ce_a_swapped, ce_b')
    files = [path]
    for name, spec in meta['components'].items():
        allowed_fit, allowed_stop = ([6, 7], [3]) if name == 'graph' else ([0, 1, 8], [9])
        if sorted(spec.get('fit_folds', [])) != allowed_fit or sorted(spec.get('selection_folds', [])) != allowed_stop:
            raise ValueError(f'{name}: incompatible fit/selection folds; refuse final R14 fusion or a CE fitted on fold 3/4')
        if spec.get('score_kind') not in ('probability', 'logit') or not spec.get('score_column'):
            raise ValueError(f'{name}: declare probability/logit and original score_column')
        if set(spec.get('files', {})) != {'fold3', 'fold4', 'test'}:
            raise ValueError(f'{name}: need existing fold3, fold4 and test files')
        files += [(path.parent / spec['files'][k]).resolve() for k in ('fold3', 'fold4', 'test')]
    if not meta.get('norm_dir') and not meta.get('train_source1_ids'):
        raise ValueError('Need the ORIGINAL R14 train Source1 ID row order to verify training-fold compatibility')
    if meta.get('train_source1_ids'):
        files.append((path.parent / meta['train_source1_ids']).resolve())
    if meta.get('norm_dir'):
        norm = (path.parent / meta['norm_dir']).resolve()
        files += [norm / f'{split}_source{i}.parquet' for split in ('train', 'test') for i in (1, 2, 3)]
    fingerprint(files)  # Missing components fail before model startup.
    return meta, list(dict.fromkeys(files))


def check_fold_mapping(args, meta):
    """A score join alone cannot prove the donor's training labels avoid 3B/4."""
    import polars as pl
    from er_v2.folds import fold_expr
    if meta.get('norm_dir'):
        donor_path = args.r14_provenance.parent / meta['norm_dir'] / 'train_source1.parquet'
    else:
        donor_path = args.r14_provenance.parent / meta['train_source1_ids']
    donor = pl.read_parquet(donor_path, columns=['entity_id']).with_row_index('sidx').with_columns(fold_expr()).rename({'fold': 'donor_fold'})
    base = pl.read_parquet(args.base_work / 'norm/train_source1.parquet', columns=['idx', 'entity_id']).rename({'idx': 'sidx'}).with_columns(fold_expr())
    if donor['entity_id'].null_count() or donor['entity_id'].n_unique() != len(donor) or len(donor) != len(base):
        raise ValueError('Original R14 training ID mapping is missing/duplicated/incomplete')
    aligned = base.join(donor.select('entity_id', 'donor_fold'), on='entity_id', how='left', validate='1:1')
    if aligned['donor_fold'].null_count() or not aligned['fold'].eq(aligned['donor_fold']).all():
        raise ValueError('R14 and R12 training folds differ after ID alignment; supplied CE provenance cannot preserve the gate')


def input_paths(args):
    _, donor = donor_files(args.r14_provenance)
    base = [args.base_work / n for n in ('run.json', 'result.json', 'selection.json', 'metrics.json')]
    base += [args.base_work / 'norm' / f'{split}_source{i}.parquet' for split in ('train', 'test') for i in (1, 2, 3)]
    selection = json.loads((args.base_work / 'selection.json').read_text())
    if selection['selected'] == 'reference':
        base += [args.base_work / 'models/stage3_metrics.json']
    exports = [args.r12_export / n for n in ('selection.json', 'metrics.json', 'r12_fold3.parquet', 'r12_fold4.parquet', 'r12_test.parquet')]
    data = [args.dataset / split / f'{split}_source{i}.tsv' for split in ('train', 'test') for i in (1, 2, 3)]
    return base + exports + donor + data + [args.dataset / 'train/train_ground_truth.tsv', args.validator]


def preflight(args):
    # Check the deadline and cheap provenance before importing model libraries.
    if remaining_seconds(args) < 10 * 60:
        raise TimeoutError('Less than ten minutes remain after the download buffer; do not start this experiment')
    paths = input_paths(args)
    safe_paths(args, paths)
    snapshots = fingerprint(paths)
    base = json.loads((args.base_work / 'run.json').read_text())
    result = json.loads((args.base_work / 'result.json').read_text())
    selection = json.loads((args.base_work / 'selection.json').read_text())
    if base.get('status') != 'complete' or result.get('status') != 'complete' or result.get('official_validation') != 'PASS':
        raise ValueError('Base must be the completed, officially validated ORIGINAL R12 run')
    if selection.get('version') != 'r10-ce-ann-1' or selection.get('selected') not in ('r10', 'reference'):
        raise ValueError('Unexpected R12 selection version')
    if any(not result.get('output_sha256', {}).get(n) for n in ('matching_results.tsv', 'candidate_pairs.tsv')):
        raise ValueError('Original R12 output hashes are required for byte-identical fallback verification')
    for name in ('selection.json', 'metrics.json'):
        if json.loads((args.r12_export / name).read_text()) != json.loads((args.base_work / name).read_text()):
            raise ValueError('R12 export metadata differs from the completed parent: ' + name)
    versions = {name: importlib.metadata.version(name) for name in ('polars', 'numpy', 'xgboost', 'pyarrow')}
    identity = base['identity']
    if versions['polars'] != identity['versions']['polars']:
        raise ValueError('Use the original .venv-r12: Polars version controls frozen fold membership')
    if identity['data'] != fingerprint([args.dataset / 'train', args.dataset / 'test']):
        raise ValueError('Dataset paths/content metadata differ from original R12')
    if identity['validator'] != fingerprint([args.validator]):
        raise ValueError('Official validator differs from original R12')
    import polars as pl
    meta, _ = donor_files(args.r14_provenance)
    check_fold_mapping(args, meta)
    for spec in meta['components'].values():
        for filename in spec['files'].values():
            columns = set(pl.scan_parquet(args.r14_provenance.parent / filename).collect_schema().names())
            if spec['score_column'] not in columns:
                raise ValueError('Declared donor score column missing: ' + filename)
            if not ({'sid', 'tid'} <= columns or {'source1_entity_id', 'target_entity_id'} <= columns or (meta.get('norm_dir') and {'sidx', 'tidx'} <= columns)):
                raise ValueError('Donor needs entity IDs or its original six norm mappings: ' + filename)
    if args.device == 'cuda':
        import numpy as np
        import xgboost as xgb
        probe = xgb.train({'tree_method': 'hist', 'device': 'cuda', 'nthread': args.threads}, xgb.DMatrix(np.eye(4), label=[0, 1, 0, 1]), 1)
        if json.loads(probe.save_config())['learner']['generic_param']['device'] == 'cpu':
            raise RuntimeError('Existing environment lacks CUDA XGBoost; no packages will be installed')
    return versions, paths, snapshots


def commands(args):
    shared = []
    for name in ('base_work', 'r12_export', 'r14_provenance', 'dataset', 'work', 'output', 'device', 'threads', 'rounds'):
        shared += ['--' + name.replace('_', '-'), str(getattr(args, name))]
    stages = {}
    for stage, names in [('select', ['selection.json', 'fusion.json']), ('evaluate', ['metrics.json']), ('inference', ['inference.json'])]:
        outputs = [args.work / n for n in names]
        if stage == 'inference':
            outputs += [args.output / n for n in ('matching_results.tsv', 'candidate_pairs.tsv')]
        stages[stage] = ([sys.executable, '-u', '-m', 'er_v2.r16_fast_fusion', stage, *shared], outputs)
    stages['validate'] = ([sys.executable, str(args.validator), '--matching', str(args.output / 'matching_results.tsv'),
        '--candidate', str(args.output / 'candidate_pairs.tsv'), '--test-dir', str(args.dataset / 'test'), '--check-ids'], [])
    return stages


def main():
    p = parser()
    args = p.parse_args()
    if not 1 <= args.threads <= 32 or not 1 <= args.rounds <= 400 or not 0 < args.max_minutes <= 60 or args.download_buffer_minutes < 10 or args.reserve_gb < 1:
        p.error('Invalid limits: threads<=32, rounds<=400, wall cap<=60min, download buffer>=10min, reserve>=1GiB')
    for name in ('base_work', 'r12_export', 'r14_provenance', 'dataset', 'validator', 'work', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    safe_paths(args)
    if args.plan:
        for stage, (command, _) in commands(args).items():
            print(stage, subprocess.list2cmdline(command))
        return
    versions, inputs, snapshots = preflight(args)
    if args.preflight:
        print(json.dumps({'status': 'preflight PASS', 'version': VERSION, 'versions': versions,
                          'seconds_available': remaining_seconds(args)}, indent=2))
        return
    args.work.mkdir(parents=True, exist_ok=True)
    import filelock
    with filelock.FileLock(str(args.work / 'run.lock'), timeout=0):
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                  if k not in ('resume', 'plan', 'preflight', 'max_minutes', 'shutdown_at', 'download_buffer_minutes')}
        identity = {'version': VERSION, 'code': code_hash(), 'runner': file_hash(Path(__file__)),
                    'settings': config, 'inputs': snapshots, 'versions': versions}
        manifest = args.work / 'run.json'
        if args.resume:
            state = json.loads(manifest.read_text())
            if state['identity'] != identity:
                raise ValueError('Inputs/code/settings changed: use a new work and output directory')
        else:
            if manifest.exists() or args.output.exists() or any(args.work.glob('*.json')):
                raise ValueError('Existing artifacts: use checked --resume or fresh work/output directories')
            state = {'identity': identity, 'started': now(), 'completed': {}}
        (args.work / 'logs').mkdir(exist_ok=True)
        state.update(status='running', pid=os.getpid())
        deadline = time.monotonic() + remaining_seconds(args)
        env = {**os.environ, 'PYTHONPATH': str(ROOT / 'code/business_entity_resolution/src') + os.pathsep + os.environ.get('PYTHONPATH', ''),
               'POLARS_MAX_THREADS': str(args.threads), 'OMP_NUM_THREADS': str(args.threads), 'OPENBLAS_NUM_THREADS': '1',
               'MKL_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUTF8': '1'}
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Interrupted by {signum}')
        previous_sigterm = signal.signal(signal.SIGTERM, interrupted)
        try:
            for stage, (command, outputs) in commands(args).items():
                if fingerprint(inputs) != snapshots:
                    raise ValueError('Original inputs changed during this run; refusing mixed evidence')
                previous = state['completed'].get(stage)
                if previous:
                    if previous['command'] != command or previous['outputs'] != fingerprint(outputs):
                        raise ValueError('Completed stage changed: ' + stage)
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError('R16 wall/download deadline reached')
                if shutil.disk_usage(args.work).free < args.reserve_gb * 1024**3:
                    raise RuntimeError('Free disk below reserve')
                logpath = args.work / 'logs' / f'{stage}.log'
                state.update(stage=stage, stage_started=now(), log=str(logpath))
                save(manifest, state)
                print(f'{now()} {stage} -> {logpath}', flush=True)
                started = time.monotonic()
                with logpath.open('w') as log:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=os.name == 'posix')
                    try:
                        while child.poll() is None:
                            time.sleep(1)
                            if time.monotonic() >= deadline or remaining_seconds(args) <= 0:
                                raise TimeoutError(stage + ': R16 wall/download deadline reached')
                            if shutil.disk_usage(args.work).free < args.reserve_gb * 1024**3:
                                raise RuntimeError('Free disk below reserve')
                    except BaseException:
                        stop_child(child)
                        raise
                if child.returncode:
                    raise RuntimeError(f'{stage} failed ({child.returncode}); inspect {logpath}')
                state['completed'][stage] = {'command': command, 'outputs': fingerprint(outputs), 'seconds': round(time.monotonic() - started, 1)}
                save(manifest, state)
            if fingerprint(inputs) != snapshots:
                raise ValueError('Original inputs changed during run')
            metrics = json.loads((args.work / 'metrics.json').read_text())
            result = {'version': VERSION, 'status': 'complete', 'official_validation': 'PASS', 'metrics': metrics,
                      'output_sha256': {n: file_hash(args.output / n) for n in ('matching_results.tsv', 'candidate_pairs.tsv')},
                      'stage_seconds': {k: v['seconds'] for k, v in state['completed'].items()}}
            save(args.work / 'result.json', result)
            selected = metrics['selected']
            (args.work / 'result.md').write_text(f'R16 CE fusion complete. Official TSV validation: PASS.\n\nSelected: {selected}. Fold-4 macro F0.5: {metrics["local_fold4"]["macro_f05"]:.9f}. Website: unmeasured.\n\n' +
                ('Exact R12 fallback: do not resubmit it.\n' if selected == 'r12' else 'New proposal passed the frozen gate; inspect selection.json before submission.\n'))
            state.update(status='complete', finished=now())
            save(manifest, state)
        except BaseException as exc:
            state.update(status='failed', error=str(exc), finished=now())
            save(manifest, state)
            save(args.work / 'result.json', {'status': 'failed', 'error': str(exc), 'finished': now()})
            raise
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == '__main__':
    main()
