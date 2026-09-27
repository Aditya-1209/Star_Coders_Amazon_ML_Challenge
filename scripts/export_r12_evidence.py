#!/usr/bin/env python3
"""Export frozen R12 evidence with entity IDs; never fit or tune a model."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code/business_entity_resolution/src'))


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def mappings(work, split):
    import numpy as np
    import polars as pl
    from er_v2.folds import fold_expr
    from er_v2.r10 import half
    frames, offset = [], 0
    for side in (1, 2, 3):
        records = pl.read_parquet(work / f'norm/{split}_source{side}.parquet',
                                  columns=['idx', 'entity_id', 'country'])
        if not np.array_equal(records['idx'].to_numpy(), np.arange(len(records))):
            raise ValueError(f'{split} source {side}: indices do not match normalized row order')
        if (records['entity_id'].null_count() or records['entity_id'].n_unique() != len(records)
                or records['entity_id'].str.len_chars().min() == 0):
            raise ValueError(f'{split} source {side}: missing or duplicate entity IDs')
        records = records.with_columns(
            split=pl.lit(split), source=pl.lit(side, pl.Int8),
            pair_idx=(pl.col('idx') + (offset if side == 3 else 0)).cast(pl.UInt32))
        if side == 2:
            offset = len(records)
        frames.append(records)
    anchors = frames[0].rename({'pair_idx': 'sidx'}).with_columns(
        subset=pl.lit('test', pl.String))
    if split == 'train':
        anchors = anchors.with_columns(fold_expr()).with_columns(
            subset=pl.when(pl.col('fold') == 3)
                .then(pl.when(half() == 0).then(pl.lit('3A')).otherwise(pl.lit('3B')))
                .otherwise(pl.concat_str(pl.lit('fold'), pl.col('fold'))))
    targets = pl.concat(frames[1:]).rename({'pair_idx': 'tidx'})
    if targets['entity_id'].n_unique() != len(targets):
        raise ValueError('Target entity IDs must be unique across Source 2 and Source 3')
    return anchors, targets


def canonical(rows, anchors, targets):
    """Left joins plus explicit checks preserve every input pair, including truth misses."""
    import polars as pl
    keys = ['sidx', 'tidx']
    if rows.select(keys).null_count().row(0) != (0, 0) or rows.select(keys).n_unique() != len(rows):
        raise ValueError('Null or duplicate pair keys')
    if 'score' in rows.columns:
        if rows['score'].null_count() or not rows['score'].is_finite().all() or not rows['score'].is_between(0, 1).all():
            raise ValueError('Invalid final confidence scores')
    result = (rows.join(anchors.select('sidx', 'split', 'subset',
                       source1_entity_id=pl.col('entity_id')), on='sidx', how='left', validate='m:1')
        .join(targets.select('tidx', target_source=pl.col('source'), target_entity_id=pl.col('entity_id')),
              on='tidx', how='left', validate='m:1'))
    if result['source1_entity_id'].null_count() or result['target_entity_id'].null_count():
        raise ValueError('Pair index has no entity-ID mapping; refusing to drop rows')
    return result


def check_metrics(actual, expected, name):
    for key in ('anchors', 'macro_f05', 'pair_precision', 'pair_recall'):
        if (key not in expected or actual.get(key) is None or not math.isfinite(actual[key])
                or not math.isfinite(expected[key]) or abs(actual[key] - expected[key]) > 1e-9):
            raise ValueError(f'{name}: frozen R12 metric mismatch for {key}: {actual.get(key)} vs {expected.get(key)}')


def snapshot(paths):
    return {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in paths}


def preflight(args):
    state, result = (read_json(args.base_work / name) for name in ('run.json', 'result.json'))
    selection = read_json(args.base_work / 'selection.json')
    if state.get('status') != 'complete' or result.get('status') != 'complete' or result.get('official_validation') != 'PASS':
        raise ValueError('Export requires a completed, officially validated R12 run')
    if selection.get('version') != 'r10-ce-ann-1' or selection.get('selected') not in ('r10', 'reference'):
        raise ValueError('Expected original R12 selection, not R13/R16 or a fast-run directory')
    if result['metrics']['selection'] != selection:
        raise ValueError('Saved selection differs from the completed result')
    versions = {name: importlib.metadata.version(name) for name in ('polars', 'numpy', 'xgboost', 'pyarrow')}
    if any(state['identity']['versions'].get(name) != value for name, value in versions.items()):
        raise ValueError('Use the original R12 environment: dependency versions affect folds and predictions')
    metadata = ['run.json', 'result.json', 'selection.json', 'metrics.json', 'r10_models.json',
                'models/stage3_metrics.json', 'ce_model/training.json', 'neural_e5/finetune.json']
    inputs = [args.base_work / name for name in metadata + ['models/r10_pair.json', 'models/r10_business.json',
        'eval_preds_stage3.parquet', 'stage3_train.parquet', 'ce_train.parquet', 'r10_test_predictions.parquet']]
    inputs += [args.base_work / f'norm/{split}_source{side}.parquet'
               for split in ('train', 'test') for side in (1, 2, 3)]
    recorded = {item['path']: item for stage in state['completed'].values() for item in stage['outputs']}
    for path in inputs:
        info = path.stat()
        previous = recorded.get(str(path))
        if path.name not in ('run.json', 'result.json') and (previous is None or
                (info.st_size, info.st_mtime_ns) != (previous['bytes'], previous['mtime_ns'])):
            raise ValueError(f'Parent artifact no longer matches completed run: {path}')
    # Preserve exact original dataset provenance, including labels used for the audit.
    for item in state['identity']['data']:
        path = Path(item['path'])
        if not path.is_relative_to(args.dataset):
            raise ValueError('Use the original R12 dataset path')
        inputs.append(path)
        if (path.stat().st_size, path.stat().st_mtime_ns) != (item['bytes'], item['mtime_ns']):
            raise ValueError(f'Dataset changed since R12: {path}')
    if read_json(args.base_work / 'metrics.json') != result['metrics']:
        raise ValueError('Metrics differ from the completed result')
    return state, result, selection, versions, metadata, inputs


def export(args):
    import polars as pl
    from er_v2 import r10
    from er_v2.metrics import macro_f05, by_country
    from er_v2.predict import write_lists
    state, result, selection, versions, metadata, inputs = preflight(args)
    before = snapshot(inputs)
    args.output.mkdir(parents=True, exist_ok=False)
    # A failed export has no manifest.json and must never be treated as complete.
    (args.output / '_INCOMPLETE').write_text('Export validation has not completed.\n')
    for name in metadata:
        path = args.output / 'metadata' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.base_work / name, path)
    parent = SimpleNamespace(work=args.base_work, dataset=args.dataset, device=args.device,
                             threads=args.threads, batch_rows=50_000, r13_features=False)
    checks = {}
    for split in ('train', 'test'):
        anchors, targets = mappings(args.base_work, split)
        anchors.write_parquet(args.output / f'{split}_source1_ids.parquet', compression='zstd')
        targets.write_parquet(args.output / f'{split}_target_ids.parquet', compression='zstd')
        for fold in ([3, 4] if split == 'train' else [None]):
            print(f'Exporting {split} fold {fold}: frozen final R12 scores', flush=True)
            subset = anchors.filter(pl.col('fold') == fold) if fold is not None else anchors
            scope = subset.select('sidx', 'country')
            if split == 'train':
                loaded, scores = r10.chosen_scores(parent, split, fold)
                if loaded != selection:
                    raise ValueError('Parent selection changed during export')
            else:
                # This is the final fused cache written by r10.inference. Its
                # completed-run fingerprint and both official TSV hashes are checked.
                scores = pl.read_parquet(args.base_work / 'r10_test_predictions.parquet').select('sidx', 'tidx', 'score')
            canonical(scores, subset, targets)  # fail before decision ownership can hide bad keys
            decisions = r10.final_decision(parent, selection, scores, scope)
            if decisions['tidx'].n_unique() != len(decisions):
                raise ValueError('Frozen decisions violate exclusive target ownership')
            if split == 'train':
                truth = r10.truth(parent, [fold])
                for name in subset['subset'].unique().sort():
                    current = subset.filter(pl.col('subset') == name)
                    # Historical 3A/3B ownership is resolved independently.
                    scoped_scores = scores.join(current.select('sidx'), on='sidx', how='semi')
                    scoped_decisions = r10.final_decision(parent, selection, scoped_scores, current.select('sidx', 'country'))
                    measured = macro_f05(scoped_decisions, truth, current['sidx'])
                    if name == 'fold4':
                        check_metrics(measured, result['metrics']['local_fold4'], name)
                        for c, value in by_country(scoped_decisions, truth, current.select('sidx', 'country')).items():
                            check_metrics(value, result['metrics']['by_country'][c], f'fold4/{c}')
                    elif name == '3A' and selection['selected'] == 'r10':
                        if abs(measured['macro_f05'] - selection['proposal']['fold3A']) > 1e-9:
                            raise ValueError('3A final fused score does not reproduce the selected R12 model')
                    elif name == '3B':
                        expected = selection['gate']['candidate_by_country' if selection['selected'] == 'r10' else 'baseline_by_country']
                        for c, value in by_country(scoped_decisions, truth, current.select('sidx', 'country')).items():
                            check_metrics(value, expected[c], f'3B/{c}')
                    checks[name] = measured
                    rows = scoped_scores.join(scoped_decisions.select('sidx', 'tidx').with_columns(selected=pl.lit(True)),
                        on=['sidx', 'tidx'], how='left', validate='1:1').with_columns(pl.col('selected').fill_null(False))
                    canonical(rows, current, targets).write_parquet(args.output / f'{name}_scores.parquet', compression='zstd')
                    canonical(truth.join(current.select('sidx'), on='sidx', how='semi'), current, targets).write_parquet(
                        args.output / f'{name}_truth.parquet', compression='zstd')
            else:
                with TemporaryDirectory(dir=args.output) as temporary:
                    for filename, rows, column in [('matching_results.tsv', decisions, 'matched_entity_ids'),
                                                   ('candidate_pairs.tsv', scores, 'candidate_entity_ids')]:
                        path = Path(temporary) / filename
                        write_lists(path, anchors.select('idx', 'entity_id'), rows, targets['entity_id'], column)
                        if digest(path) != result['output_sha256'][filename]:
                            raise ValueError(f'Final test scores do not reproduce the officially validated {filename}')
                scored = scores.join(decisions.select('sidx', 'tidx').with_columns(selected=pl.lit(True)),
                    on=['sidx', 'tidx'], how='left', validate='1:1').with_columns(pl.col('selected').fill_null(False))
                canonical(scored, anchors, targets).write_parquet(args.output / 'test_scores.parquet', compression='zstd')
                del scored
                checks['test'] = {'official_tsv_hashes_match': True, 'anchors': len(anchors),
                                  'candidates': len(scores), 'matches': len(decisions)}
            del scores, decisions
            gc.collect()
    if snapshot(inputs) != before:
        raise ValueError('Parent evidence changed during export')
    outputs = {str(path.relative_to(args.output)): {'bytes': path.stat().st_size, 'sha256': digest(path)}
               for path in sorted(args.output.rglob('*')) if path.is_file() and path.name != '_INCOMPLETE'}
    save_json(args.output / 'manifest.json', {'version': 'r12-evidence-export-1', 'status': 'complete',
        'selection': selection, 'checks': checks, 'versions': versions, 'parent_code': state['identity']['code'],
        'exporter_sha256': digest(Path(__file__)), 'files': outputs,
        'join_keys': ['split', 'source1_entity_id', 'target_source', 'target_entity_id'],
        'protocol': 'Tune on 3A; gate once on 3B; fold4 is report-only and has been inspected before. '
                    'Use a full candidate union with explicit missing scores; never align independent runs on sidx/tidx.'})
    (args.output / '_INCOMPLETE').unlink()
    print(f'Complete: {args.output / "manifest.json"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base-work', 'dataset', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--threads', type=int, default=30)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('Threads must be positive')
    for name in ('base_work', 'dataset', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    if any(args.output == path or args.output.is_relative_to(path) or path.is_relative_to(args.output)
           for path in (args.base_work, args.dataset)):
        parser.error('Export must be separate from parent artifacts and dataset')
    for name in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS'):
        os.environ[name] = str(args.threads)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    export(args)


if __name__ == '__main__':
    main()
