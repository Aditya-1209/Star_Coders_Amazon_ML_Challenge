"""Deadline-sized R16: reuse R12 neural evidence and learn business presence.

Fit 6/7, early stop/select 3A, unchanged one-shot .0005 gate on 3B against
actual completed R12, report fold 4 only after selection. No neural training.
"""
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
from .folds import fold_expr
from .metrics import macro_f05
from .r10_retrieval import save_json
from .train import PARAMS, fit, predict_frame

VERSION = 'r16-fast-r12-presence-1'
PRESENCE_INPUTS = ['ce_logit', 'p2', 'ncos', 'core_tset', 'addr_tset',
    'addr_len_r', 'name_num_conflict', 'r15_sibling_ce',
    'r15_sibling_name_tset', 'r15_sibling_address_tset',
    'r15_sibling_number_overlap', 'r15_ce_alternative_margin']
parent, frame_for, reference = r15.parent, r15.frame_for, r15.reference
prepare, audit = r15.prepare, r15.audit


def business_frame(frame, anchors):
    """Label-free aggregates of earlier, out-of-fold signals; all businesses.

    No candidate counts, IDF, country/record IDs or fitted final predictions.
    Missing evidence is represented explicitly; it never becomes agreement.
    """
    if anchors['sidx'].n_unique() != len(anchors):
        raise ValueError('Duplicate business anchors')
    data = frame.select('sidx', pl.col(PRESENCE_INPUTS).cast(pl.Float32))
    for name in PRESENCE_INPUTS:
        if data.filter(pl.col(name).is_not_null() & ~pl.col(name).is_finite()).height:
            raise ValueError('Non-finite presence evidence: ' + name)
    expressions = [pl.lit(1., pl.Float32).alias('has_candidates')]
    for name in PRESENCE_INPUTS:
        expressions.extend([pl.col(name).max().alias(name + '_max'),
                            pl.col(name).mean().alias(name + '_mean'),
                            pl.col(name).is_null().mean().cast(pl.Float32).alias(name + '_missing')])
    grouped = data.group_by('sidx').agg(expressions)
    top = data.group_by('sidx').agg(pl.col('ce_logit').sort(descending=True).head(2).alias('_top'))
    top = top.select('sidx', ce_top_gap=(pl.col('_top').list.first()
        - pl.col('_top').list.get(1, null_on_oob=True).fill_null(pl.col('_top').list.first())))
    return (anchors.select('sidx').join(grouped, on='sidx', how='left', validate='1:1')
        .join(top, on='sidx', how='left', validate='1:1')
        .with_columns(pl.exclude('sidx').fill_null(0.)).sort('sidx').with_columns(fold_expr()))


def presence_labels(args, business):
    # Full organizer truth: an unretrieved true match is still positive.
    positive = r10.truth(parent(args), [6, 7, 3]).select('sidx').unique().with_columns(label=pl.lit(1, pl.Int8))
    return business.join(positive, on='sidx', how='left').with_columns(pl.col('label').fill_null(0))


def fit_final(args):
    # Two enhanced pair models, not a rebuilt historical reference ensemble.
    r15.fit_final(args)
    data = frame_for(args, 'train', [6, 7, 3])
    country = r10.anchors(parent(args)).filter(fold_expr().is_in([6, 7, 3]))
    business = presence_labels(args, business_frame(data, country))
    del data
    gc.collect()
    tr = business.filter(pl.col('fold').is_in([6, 7]))
    va = business.filter((pl.col('fold') == 3) & (r10.half() == 0))
    features = [c for c in business.columns if c not in ('sidx', 'fold', 'label')]
    if tr.is_empty() or va.is_empty():
        raise ValueError('Presence model needs train and validation businesses')
    path = args.work / 'models/presence.json'
    if tr['label'].n_unique() < 2:
        meta = {'features': features, 'constant': float(tr['label'].mean())}
        save_json(path, meta)
    else:
        model = fit(tr, features, va, args.rounds,
            {**PARAMS, 'max_depth': 4, 'eta': .04, 'reg_lambda': 10., 'seed': 1616,
             'device': args.device, 'nthread': args.threads, 'max_cached_hist_node': 256})
        model.save_model(path)
        meta = {'features': features, 'best_iteration': model.best_iteration}
    info = json.loads((args.work / 'models.json').read_text())
    info.update(version=VERSION, presence=meta,
        presence_protocol={'train_folds': [6, 7], 'early_stopping': '3A',
                           'labels': 'full organizer truth', 'uses_fitted_pair_predictions': False})
    save_json(args.work / 'models.json', info)


def predict_presence(args, frame, country):
    meta = json.loads((args.work / 'models.json').read_text())['presence']
    business = business_frame(frame, country)
    if 'constant' in meta:
        values = np.full(len(business), meta['constant'], np.float32)
    else:
        model = xgb.Booster(model_file=str(args.work / 'models/presence.json'))
        model.set_param({'device': args.device, 'nthread': args.threads})
        values = predict_frame(model, business, meta['features'], args.batch_rows)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Invalid presence probabilities')
    return business.select('sidx').with_columns(presence=pl.Series(values))


def eligible(scores, presence, cutoff):
    if not 0 <= cutoff <= 1:
        raise ValueError('Invalid presence cutoff')
    joined = scores.join(presence, on='sidx', how='left', validate='m:1')
    if joined['presence'].null_count() or not joined['presence'].is_finite().all():
        raise ValueError('Missing/non-finite presence probabilities')
    # Filter before exclusive target ownership, not after stealing a target.
    return joined.filter(pl.col('presence') >= cutoff).select('sidx', 'tidx', 'score')


def select(args):
    p = parent(args)
    country = r10.anchors(p).filter(fold_expr() == 3)
    a, b = country.filter(r10.half() == 0), country.filter(r10.half() == 1)
    target = r10.truth(p, [3])
    frame = frame_for(args, 'train', [3])
    base_selection, base = reference(args, 'train', 3)
    presence = predict_presence(args, frame, country)
    trials, best, chosen = [], None, None
    for kind in ('reference', *r10.VARIANTS, 'mean'):
        current = base if kind == 'reference' else r15.predict(args, frame, kind)
        for weight in ((0.,) if kind == 'reference' else (.5, 1.)):
            scores = base if kind == 'reference' else r10.blend(current, base, weight)
            for cutoff in (0., .25, .5, .75, .9, .95):
                allowed = eligible(scores.join(a.select('sidx'), on='sidx', how='semi'), presence, cutoff)
                _, threshold = tune_threshold(allowed, target, a['sidx'], 'score')
                cutoffs = tune_country_thresholds(allowed, target, a, threshold, 'score')
                value = macro_f05(decide_country(allowed, threshold, a, cutoffs, 'score'), target, a['sidx'])['macro_f05']
                trial = {'model': kind, 'weight': weight, 'presence_cutoff': cutoff,
                         'threshold': threshold, 'country_thresholds': cutoffs, 'fold3A': value}
                trials.append(trial)
                if best is None or value > best['fold3A'] + 1e-12:
                    best, chosen = trial, scores
    allowed = eligible(chosen.join(b.select('sidx'), on='sidx', how='semi'), presence, best['presence_cutoff'])
    proposed = decide_country(allowed, best['threshold'], b, best['country_thresholds'], 'score')
    baseline = r10.final_decision(p, base_selection, base.join(b.select('sidx'), on='sidx', how='semi'), b)
    comparison = r10.gate(proposed, baseline, target, b, args.minimum_gain)
    selection = {'version': VERSION, 'selected': 'r16_fast' if comparison['passed'] else 'r12',
        'proposal': best, 'gate': comparison, 'trials_on_3A': trials,
        'baseline': 'actual completed R12 selection', 'base_selection': base_selection,
        'protocol_note': 'Fit 6/7; early stop/select 3A; unchanged .0005 one-shot gate on 3B. '
                         'Inherited models used fold 3. Fold 4 is report-only and previously inspected.'}
    save_json(args.work / 'selection.json', selection)
    print(json.dumps(selection, indent=2), flush=True)


def chosen(args, split, fold=None):
    selection = json.loads((args.work / 'selection.json').read_text())
    _, base = reference(args, split, fold)
    if selection['selected'] == 'r12':
        return selection, base
    frame = frame_for(args, split, [fold] if fold is not None else None)
    country = r10.anchors(parent(args), split)
    if fold is not None:
        country = country.filter(fold_expr() == fold)
    proposal = selection['proposal']
    scores = base if proposal['model'] == 'reference' else r10.blend(
        r15.predict(args, frame, proposal['model']), base, proposal['weight'])
    return selection, scores.join(predict_presence(args, frame, country), on='sidx', how='left', validate='m:1')


def decide(args, selection, scores, country):
    if selection['selected'] == 'r12':
        return r15.decide(args, selection, scores, country)
    proposal = selection['proposal']
    allowed = eligible(scores.select('sidx', 'tidx', 'score'),
                       scores.select('sidx', 'presence').unique(), proposal['presence_cutoff'])
    return decide_country(allowed, proposal['threshold'], country, proposal['country_thresholds'], 'score')


def evaluate(args):
    # Reuse the tested report writer with this module's decision functions.
    selection, scores = chosen(args, 'train', 4)
    p = parent(args)
    country = r10.anchors(p).filter(fold_expr() == 4)
    pred = decide(args, selection, scores, country)
    base_selection, base = reference(args, 'train', 4)
    baseline = r10.final_decision(p, base_selection, base, country)
    target = r10.truth(p, [4])
    from .metrics import by_country
    singleton = country.join(target.select('sidx').unique(), on='sidx', how='anti')
    report = {'version': VERSION, 'selected': selection['selected'],
        'local_fold4': macro_f05(pred, target, country['sidx']),
        'reference_fold4': macro_f05(baseline, target, country['sidx']),
        'by_country': by_country(pred, target, country), 'selection': selection,
        'singletons': {'businesses': len(singleton), 'false_matched_businesses':
            pred.select('sidx').unique().join(singleton.select('sidx'), on='sidx', how='semi').height},
        'errors': r15.error_report(args, scores, pred, target, country),
        'target_leaderboard': .99, 'amazon_score': None, 'user_reported_r12_leaderboard': .984,
        'limitations': ['France has no labels; website score is unmeasured.',
            'Reuses R12 neural models/candidates; retrieval ceiling is unchanged.', selection['protocol_note']]}
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
    scores.select('sidx', 'tidx', 'score').write_parquet(args.work / 'test_predictions.parquet')
    print(f'Wrote submission; selected {selection["selected"]}. Validation is next.', flush=True)


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
        p.error('Positive counts required; original .0005 gate is fixed')
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
