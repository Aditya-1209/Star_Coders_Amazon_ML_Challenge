"""R15: reuse completed R12 neural evidence; gate a new classifier against R12.

No neural optimizer runs here. Fit folds 6/7, early stopping/choice 3A,
unchanged one-shot gate on 3B, report fold 4 only after selection is saved.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import xgboost as xgb

from . import r10
from .decision import decide_country, tune_threshold, tune_country_thresholds
from .features import feature_names
from .folds import fold_expr
from .graph import hop_keep
from .metrics import macro_f05, by_country
from .r10_retrieval import save_json
from .r15_features import enrich
from .run_block import load_split, parquet_rows, split_countries
from .train import PARAMS, fit, predict_frame

VERSION = 'r15-ce-cross-source-1'
# These see only graph folds at train time, but every business at test time.
# Do not amplify the population mismatch with new target-owner statistics.
INCOMPLETE_CONTEXT = {'ce_rank_t', 'ce_gap_t', 'ncos_rank_t', 'ncos_gap_t'}
VARIANTS = r10.VARIANTS


def parent(args):
    return SimpleNamespace(**{**vars(args), 'work': args.base_work})


def frame_for(args, split, folds=None):
    data = pl.scan_parquet(args.work / f'features_{split}.parquet')
    if folds is not None:
        data = data.filter(pl.col('fold').is_in(folds))
    return data.collect(engine='streaming')


def prepare(args):
    split = args.split
    output = args.work / f'features_{split}.parquet'
    temporary = output.with_suffix('.partial.parquet')
    writer = None
    n_s2 = parquet_rows(args.base_work / f'norm/{split}_source2.parquet')
    try:
        for country in split_countries(args.base_work, split):
            s1, targets = load_split(args.base_work, split, country, ['idx', 'core_n', 'addr_n'])
            ids = s1.select(sidx=pl.col('idx').cast(pl.UInt32))
            source = (pl.scan_parquet(args.base_work / f'stage3_{split}.parquet').filter(hop_keep())
                      .join(ids.lazy(), on='sidx', how='semi').collect(engine='streaming'))
            scores = (pl.scan_parquet(args.base_work / f'ce_{split}.parquet')
                      .join(ids.lazy(), on='sidx', how='semi').collect(engine='streaming'))
            data = r10.attach_ce(source, scores)
            del source, scores, s1, ids
            print(f'{split} {country}: {len(data):,} pairs; building CE cross-source evidence', flush=True)
            progress = lambda done, total: print(f'  {country}: {done:,}/{total:,} pairs', flush=True)
            data = enrich(data, targets, n_s2, args.threads, progress)
            if writer is None:
                writer = pq.ParquetWriter(temporary, data.to_arrow().schema, compression='zstd')
            # Bounded row groups support subsequent streaming/filter reads.
            for chunk in data.iter_slices(100_000):
                writer.write_table(chunk.to_arrow())
            del targets, data
            gc.collect()
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError('No R15 candidate rows')
    temporary.replace(output)


def fit_final(args):
    data = frame_for(args, 'train', [6, 7, 3])
    tr = data.filter(pl.col('fold').is_in([6, 7]))
    va = data.filter((pl.col('fold') == 3) & (r10.half() == 0))
    shift = json.loads((args.base_work / 'shift_check.json').read_text())
    features = [c for c in feature_names(data, 'enhanced' if shift.get('allow_enhanced') else 'baseline')
                if c not in INCOMPLETE_CONTEXT]
    del data
    info = {'version': VERSION, 'features': features, 'variants': {},
            'train_folds': [6, 7], 'early_stopping': '3A',
            'excluded_incomplete_context': sorted(INCOMPLETE_CONTEXT)}
    for name, parameters in VARIANTS.items():
        model = fit(r10.weights(tr) if name == 'business' else tr, features,
                    r10.weights(va) if name == 'business' else va, args.rounds,
                    {**PARAMS, **parameters, 'device': args.device, 'nthread': args.threads,
                     'max_cached_hist_node': 1024})
        model.save_model(args.work / 'models' / f'{name}.json')
        info['variants'][name] = {'best_iteration': model.best_iteration, 'parameters': parameters}
        del model
        gc.collect()
    save_json(args.work / 'models.json', info)


def predict(args, frame, choice):
    meta = json.loads((args.work / 'models.json').read_text())
    names = list(VARIANTS) if choice == 'mean' else [choice]
    values = np.zeros(len(frame), np.float32)
    for name in names:
        model = xgb.Booster(model_file=str(args.work / 'models' / f'{name}.json'))
        model.set_param({'device': args.device, 'nthread': args.threads})
        values += predict_frame(model, frame, meta['features'], args.batch_rows) / len(names)
    return frame.select('sidx', 'tidx').with_columns(score=pl.Series(values))


def reference(args, split, fold=None):
    """The actual completed R12 selection, including its frozen threshold."""
    p = parent(args)
    if split == 'test':
        selection = json.loads((args.base_work / 'selection.json').read_text())
        scores = pl.read_parquet(args.base_work / 'r10_test_predictions.parquet')
        return selection, scores
    return r10.chosen_scores(p, split, fold)


def error_report(args, candidates, pred, target, country):
    """Aggregate diagnostics, not additional training labels or threshold tuning."""
    keys = ['sidx', 'tidx']
    false_neg = target.join(pred.select(keys), on=keys, how='anti')
    rejected = false_neg.join(candidates.select(keys), on=keys, how='semi')
    false_pos = pred.select(keys).join(target, on=keys, how='anti')
    _, targets = load_split(args.base_work, 'train', columns=['idx', 'addr_n'])
    missing = targets.select(tidx=pl.col('idx').cast(pl.UInt32),
                             missing_address=pl.col('addr_n').fill_null('') == '')
    groups = {}
    for name, rows in [('false_negative', false_neg), ('retrieved_rejected', rejected), ('false_positive', false_pos)]:
        enriched = rows.join(missing, on='tidx', how='left').join(country, on='sidx', how='left')
        groups[name] = {'pairs': len(rows), 'missing_address_pairs': int(enriched['missing_address'].sum()),
                       'by_country': enriched.group_by('country').agg(pl.len().alias('pairs'),
                           pl.col('missing_address').sum().alias('missing_address_pairs')).to_dicts()}
    groups['never_retrieved_pairs'] = len(false_neg) - len(rejected)
    values = r10.per_business(pred, target, country['sidx'])
    groups['businesses_below_perfect'] = int((values['f'] < 1).sum())
    return groups


def audit(args):
    # Audit the existing model on selection half 3A; do not use fold-4 errors
    # to fit or reselect R15. Final fold-4 breakdown is report-only.
    p = parent(args)
    country = r10.anchors(p).filter((fold_expr() == 3) & (r10.half() == 0))
    target = r10.truth(p, [3]).join(country.select('sidx'), on='sidx', how='semi')
    selection, scores = reference(args, 'train', 3)
    scores = scores.join(country.select('sidx'), on='sidx', how='semi')
    pred = r10.final_decision(p, selection, scores, country)
    report = {'split': '3A', 'baseline': 'completed R12',
              'metrics': macro_f05(pred, target, country['sidx']),
              'errors': error_report(args, scores, pred, target, country)}
    save_json(args.work / 'baseline_audit.json', report)
    print(json.dumps(report, indent=2), flush=True)


def select(args):
    p = parent(args)
    country = r10.anchors(p).filter(fold_expr() == 3)
    a, b = country.filter(r10.half() == 0), country.filter(r10.half() == 1)
    target = r10.truth(p, [3])
    frame = frame_for(args, 'train', [3])
    base_selection, base = reference(args, 'train', 3)
    trials, best, chosen = [], None, None
    for name in (*VARIANTS, 'mean'):
        current = predict(args, frame, name)
        for weight in (.25, .5, .75, 1.):
            scores = r10.blend(current, base, weight)
            _, threshold = tune_threshold(scores, target, a['sidx'], 'score')
            cutoffs = tune_country_thresholds(scores, target, a, threshold, 'score')
            value = macro_f05(decide_country(scores.join(a.select('sidx'), on='sidx', how='semi'),
                threshold, a, cutoffs, 'score'), target, a['sidx'])['macro_f05']
            entry = {'model': name, 'weight': weight, 'threshold': threshold,
                     'country_thresholds': cutoffs, 'fold3A': value}
            trials.append(entry)
            if best is None or value > best['fold3A'] + 1e-12:
                best, chosen = entry, scores
    proposed = decide_country(chosen.join(b.select('sidx'), on='sidx', how='semi'), best['threshold'],
                              b, best['country_thresholds'], 'score')
    baseline = r10.final_decision(p, base_selection, base.join(b.select('sidx'), on='sidx', how='semi'), b)
    comparison = r10.gate(proposed, baseline, target, b, args.minimum_gain)
    selection = {'version': VERSION, 'selected': 'r15' if comparison['passed'] else 'r12',
        'proposal': best, 'gate': comparison, 'trials_on_3A': trials,
        'baseline': 'completed R12 selection', 'base_selection': base_selection,
        'protocol_note': 'Same 3A/3B split and gate as R12; inherited models already used fold 3. '
                         'Fold 4 is report-only and has been inspected in earlier experiments.'}
    save_json(args.work / 'selection.json', selection)
    print(json.dumps(selection, indent=2), flush=True)


def chosen(args, split, fold=None):
    selection = json.loads((args.work / 'selection.json').read_text())
    base_selection, base = reference(args, split, fold)
    if selection['selected'] == 'r12':
        return selection, base
    data = frame_for(args, split, [fold] if fold is not None else None)
    proposal = selection['proposal']
    return selection, r10.blend(predict(args, data, proposal['model']), base, proposal['weight'])


def decide(args, selection, scores, country):
    if selection['selected'] == 'r12':
        return r10.final_decision(parent(args), selection['base_selection'], scores, country)
    p = selection['proposal']
    return decide_country(scores, p['threshold'], country, p['country_thresholds'], 'score')


def evaluate(args):
    selection, scores = chosen(args, 'train', 4)
    p = parent(args)
    country = r10.anchors(p).filter(fold_expr() == 4)
    pred = decide(args, selection, scores, country)
    base_selection, base = reference(args, 'train', 4)
    baseline = r10.final_decision(p, base_selection, base, country)
    target = r10.truth(p, [4])
    report = {'version': VERSION, 'selected': selection['selected'],
        'local_fold4': macro_f05(pred, target, country['sidx']),
        'reference_fold4': macro_f05(baseline, target, country['sidx']),
        'by_country': by_country(pred, target, country),
        'errors': error_report(args, scores, pred, target, country),
        'baseline_errors': error_report(args, base, baseline, target, country),
        'target_leaderboard': .99, 'amazon_score': None, 'user_reported_r12_leaderboard': .984,
        'selection': selection, 'limitations': ['France has no labels; local improvement does not certify leaderboard improvement.',
        'Unchanged retrieval ceiling; R15 reuses R12 neural models and candidates.']}
    save_json(args.work / 'metrics.json', report)
    print(json.dumps(report, indent=2), flush=True)


def inference(args):
    from .predict import write_lists
    selection, scores = chosen(args, 'test')
    country = r10.anchors(parent(args), 'test')
    pred = decide(args, selection, scores, country)
    s1 = pl.read_parquet(args.base_work / 'norm/test_source1.parquet', columns=['idx', 'entity_id'])
    targets = pl.concat([pl.read_parquet(args.base_work / f'norm/test_source{i}.parquet',
                        columns=['entity_id']) for i in (2, 3)])['entity_id']
    args.output.mkdir(parents=True, exist_ok=True)
    write_lists(args.output / 'candidate_pairs.tsv', s1, scores, targets, 'candidate_entity_ids')
    write_lists(args.output / 'matching_results.tsv', s1, pred, targets, 'matched_entity_ids')
    scores.write_parquet(args.work / 'test_predictions.parquet')
    print(f'Wrote submission; selected {selection["selected"]}. Official validation is next.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['audit', 'prepare', 'fit', 'select', 'evaluate', 'inference'])
    p.add_argument('--base-work', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--split', choices=['train', 'test'], default='train')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--threads', type=int, default=30)
    p.add_argument('--rounds', type=int, default=1400)
    p.add_argument('--batch-rows', type=int, default=50000)
    p.add_argument('--minimum-gain', type=float, default=.0005)
    args = p.parse_args()
    if min(args.threads, args.rounds, args.batch_rows) < 1 or args.minimum_gain != .0005:
        p.error('Positive counts required; R15 retains the original .0005 gate')
    for key in ('base_work', 'work', 'output', 'dataset'):
        setattr(args, key, getattr(args, key).resolve())
    paths = [args.base_work, args.work, args.output, args.dataset]
    if any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
           for i, a in enumerate(paths) for b in paths[i + 1:]):
        p.error('Base, work, output and dataset must be separate non-nested directories')
    args.work.mkdir(parents=True, exist_ok=True)
    (args.work / 'models').mkdir(exist_ok=True)
    {'audit': audit, 'prepare': prepare, 'fit': fit_final, 'select': select,
     'evaluate': evaluate, 'inference': inference}[args.stage](args)


if __name__ == '__main__':
    main()
