"""Missing-target-address specialist; keep the original gate against actual R12."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import polars as pl
import xgboost as xgb
from . import r10, r15
from .decision import decide_country, tune_threshold, tune_country_thresholds
from .features import feature_names
from .folds import fold_expr
from .metrics import macro_f05, by_country
from .r10_retrieval import save_json
from .run_block import parquet_rows
from .train import PARAMS, fit, predict_frame

VERSION = 'r15-missing-address-specialist-1'
parent, frame_for, reference, audit = r15.parent, r15.frame_for, r15.reference, r15.audit
VARIANTS = {'pair': {'max_depth': 6, 'eta': .04, 'reg_lambda': 10., 'seed': 1515},
            'business': {'max_depth': 6, 'eta': .04, 'reg_lambda': 12., 'seed': 2515}}


def missing_address():
    return pl.col('addr_len_r').fill_null(0) == 0


def trusted_support(frame, n_source2):
    """Donor is strongest OTHER-source CE record, not a label-selected anchor.

    Margin against that donor's own source alternatives tests ambiguity. The
    fixed logit 2.2 / margin 1 confidence rule is feature engineering, not a
    propagated match decision. Null evidence never becomes agreement.
    """
    keys = ['sidx', '_side']
    pairs = frame.select('sidx', 'tidx', 'ce_logit').with_columns(
        _side=(pl.col('tidx') >= n_source2).cast(pl.Int8))
    best = pairs.group_by(keys).agg(pl.col('ce_logit').sort(descending=True).head(2).alias('_best'))
    donor = best.select('sidx', _side=1 - pl.col('_side'),
        rescue_donor_logit=pl.col('_best').list.first(),
        rescue_donor_margin=pl.col('_best').list.first() - pl.col('_best').list.get(1, null_on_oob=True))
    result = frame.with_columns(_side=(pl.col('tidx') >= n_source2).cast(pl.Int8)).join(
        donor, on=keys, how='left', validate='m:1')
    confident = ((pl.col('rescue_donor_logit') >= 2.2)
        & (pl.col('rescue_donor_margin').is_null() | (pl.col('rescue_donor_margin') >= 1.))).fill_null(False)
    return result.with_columns(rescue_donor_confident=confident.cast(pl.Float32),
        *[pl.when(confident).then(pl.col('r15_sibling_' + name)).otherwise(None).cast(pl.Float32)
          .alias('rescue_trusted_' + name) for name in
          ('ce', 'name_ratio', 'name_tset', 'address_ratio', 'address_tset', 'number_overlap')]).drop('_side')


def prepare(args):
    r15.prepare(args)
    path = args.work / f'features_{args.split}.parquet'
    frame = pl.read_parquet(path)
    frame = trusted_support(frame, parquet_rows(args.base_work / f'norm/{args.split}_source2.parquet'))
    partial = path.with_suffix('.partial.parquet')
    frame.write_parquet(partial, compression='zstd', row_group_size=100_000)
    partial.replace(path)
    print(f'{args.split}: {len(frame):,} candidates; {frame.filter(missing_address()).height:,} missing-target-address pairs', flush=True)


def fit_final(args):
    data = frame_for(args, 'train', [6, 7, 3]).filter(missing_address())
    tr = data.filter(pl.col('fold').is_in([6, 7]))
    va = data.filter((pl.col('fold') == 3) & (r10.half() == 0))
    shift = json.loads((args.base_work / 'shift_check.json').read_text())
    features = [c for c in feature_names(data, 'enhanced' if shift.get('allow_enhanced') else 'baseline')
                if c not in r15.INCOMPLETE_CONTEXT and not c.startswith('r15_sibling_')]
    info = {'version': VERSION, 'features': features, 'variants': {}, 'train_folds': [6, 7],
        'early_stopping': '3A', 'specialist': 'missing target address only',
        'train_pairs': len(tr), 'validation_pairs': len(va),
        'excluded_incomplete_context': sorted(r15.INCOMPLETE_CONTEXT)}
    del data
    for name, params in VARIANTS.items():
        model = fit(r10.weights(tr) if name == 'business' else tr, features,
            r10.weights(va) if name == 'business' else va, args.rounds,
            {**PARAMS, **params, 'device': args.device, 'nthread': args.threads, 'max_cached_hist_node': 512})
        model.save_model(args.work / 'models' / f'{name}.json')
        info['variants'][name] = {'best_iteration': model.best_iteration, 'parameters': params}
        del model
        gc.collect()
    save_json(args.work / 'models.json', info)


def predict(args, frame, choice):
    data = frame.filter(missing_address())
    meta = json.loads((args.work / 'models.json').read_text())
    names = list(VARIANTS) if choice == 'mean' else [choice]
    values = np.zeros(len(data), np.float32)
    for name in names:
        model = xgb.Booster(model_file=str(args.work / 'models' / f'{name}.json'))
        model.set_param({'device': args.device, 'nthread': args.threads})
        values += predict_frame(model, data, meta['features'], args.batch_rows) / len(names)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite specialist predictions')
    return data.select('sidx', 'tidx').with_columns(score=pl.Series(values))


def conditional_blend(base, specialist, weight):
    """Only replace scores for specialist keys; preserve every base candidate."""
    if not 0 <= weight <= 1:
        raise ValueError('Invalid blend weight')
    keys = ['sidx', 'tidx']
    if specialist.join(base.select(keys), on=keys, how='anti').height:
        raise ValueError('Specialist candidate missing from completed R12 reference')
    return (base.join(specialist.rename({'score': '_specialist'}), on=keys, how='left', validate='1:1')
        .with_columns(score=pl.when(pl.col('_specialist').is_not_null()).then(
            weight * pl.col('_specialist') + (1 - weight) * pl.col('score')).otherwise(pl.col('score')))
        .drop('_specialist'))


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
            scores = conditional_blend(base, current, weight)
            _, threshold = tune_threshold(scores, target, a['sidx'], 'score')
            cutoffs = tune_country_thresholds(scores, target, a, threshold, 'score')
            value = macro_f05(decide_country(scores.join(a.select('sidx'), on='sidx', how='semi'),
                threshold, a, cutoffs, 'score'), target, a['sidx'])['macro_f05']
            trial = {'model': name, 'weight': weight, 'threshold': threshold,
                     'country_thresholds': cutoffs, 'fold3A': value}
            trials.append(trial)
            if best is None or value > best['fold3A'] + 1e-12:
                best, chosen = trial, scores
    proposed = decide_country(chosen.join(b.select('sidx'), on='sidx', how='semi'),
        best['threshold'], b, best['country_thresholds'], 'score')
    baseline = r10.final_decision(p, base_selection, base.join(b.select('sidx'), on='sidx', how='semi'), b)
    comparison = r10.gate(proposed, baseline, target, b, args.minimum_gain)
    result = {'version': VERSION, 'selected': 'r15_rescue' if comparison['passed'] else 'r12',
        'proposal': best, 'gate': comparison, 'trials_on_3A': trials, 'base_selection': base_selection,
        'baseline': 'actual completed R12',
        'protocol_note': 'Fit missing-address pairs on 6/7, early stop/select 3A, unchanged one-shot .0005 gate on 3B. '
                         'Previous experiments inspected 3B and fold 4; this is not a fresh nested holdout.'}
    save_json(args.work / 'selection.json', result)
    print(json.dumps(result, indent=2), flush=True)


def chosen(args, split, fold=None):
    selection = json.loads((args.work / 'selection.json').read_text())
    _, base = reference(args, split, fold)
    if selection['selected'] == 'r12':
        return selection, base
    frame = frame_for(args, split, [fold] if fold is not None else None)
    trial = selection['proposal']
    return selection, conditional_blend(base, predict(args, frame, trial['model']), trial['weight'])


decide = r15.decide


def evaluate(args):
    selection, scores = chosen(args, 'train', 4)
    p = parent(args)
    country = r10.anchors(p).filter(fold_expr() == 4)
    target = r10.truth(p, [4])
    pred = decide(args, selection, scores, country)
    base_selection, base = reference(args, 'train', 4)
    baseline = r10.final_decision(p, base_selection, base, country)
    report = {'version': VERSION, 'selected': selection['selected'],
        'local_fold4': macro_f05(pred, target, country['sidx']),
        'reference_fold4': macro_f05(baseline, target, country['sidx']),
        'by_country': by_country(pred, target, country), 'selection': selection,
        'errors': r15.error_report(args, scores, pred, target, country),
        'baseline_errors': r15.error_report(args, base, baseline, target, country),
        'target_leaderboard': .99, 'amazon_score': None, 'user_reported_r12_leaderboard': .984,
        'limitations': ['France has no labels; local gain does not establish a .99 website score.',
            'Retrieval and neural models are unchanged.', selection['protocol_note']]}
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['audit', 'prepare', 'fit', 'select', 'evaluate', 'inference'])
    for name in ('base-work', 'work', 'output', 'dataset'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--split', choices=['train', 'test'], default='train')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--threads', type=int, default=30)
    p.add_argument('--rounds', type=int, default=1400)
    p.add_argument('--batch-rows', type=int, default=50000)
    p.add_argument('--minimum-gain', type=float, default=.0005)
    args = p.parse_args()
    if min(args.threads, args.rounds, args.batch_rows) < 1 or args.minimum_gain != .0005:
        p.error('Positive counts required; original .0005 gate remains fixed')
    for name in ('base_work', 'work', 'output', 'dataset'):
        setattr(args, name, getattr(args, name).resolve())
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
