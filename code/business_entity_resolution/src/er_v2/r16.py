"""R16: neural pair ensembles plus an independently fitted no-match classifier.

Fit on folds 6/7; early stop and choose one proposal on 3A; accept/reject on
3B against a rebuilt R12-style CE ensemble; report fold 4 without reselection.
The earlier pipeline uses fold 3, so this is not a fully nested holdout.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import polars as pl
import xgboost as xgb
from . import r10
from .decision import decide_country, tune_threshold, tune_country_thresholds
from .folds import fold_expr
from .metrics import macro_f05, by_country
from .r10_retrieval import save_json
from .r13_evidence import FEATURES as SIBLING_FEATURES, INCOMPLETE_TARGET_CONTEXT
from .stage3 import feature_cols
from .train import PARAMS, fit, predict_frame

VERSION = 'r16-pair-presence-1'
KEYS = ['sidx', 'tidx']
# Explicit inputs: no IDs, labels, country codes, or final model predictions.
# These signals are out of sample for folds 6/7: stage 2 trains on 2/5 and CE
# trains on 0/1/8 (9 early stopping). Never stack fitted pair-model predictions.
PRESENCE_INPUTS = ['ce_logit', 'p2', 'ncos', 'core_tset', 'addr_tset',
                   'addr_len_r', 'name_num_conflict', 'sib_cos_max', 'sib_count']


def frame_for(args, split, folds=None):
    return r10.frame_for(SimpleNamespace(**{**vars(args), 'r13_features': True}), split, folds)


def model_features(frame, profile):
    all_features = feature_cols(frame, profile)
    return {
        'reference': [c for c in all_features if c not in SIBLING_FEATURES],
        'enhanced': [c for c in all_features if c not in INCOMPLETE_TARGET_CONTEXT],
    }


def business_frame(frame, anchors):
    """One label-free row per business, including those with zero candidates."""
    if anchors['sidx'].n_unique() != len(anchors):
        raise ValueError('Duplicate business anchors')
    if set(PRESENCE_INPUTS) - set(frame.columns):
        raise ValueError('Missing presence-model inputs')
    # Hop pairs have no stage-2 p2; zero means no direct stage-2 evidence.
    data = frame.select('sidx', pl.col(PRESENCE_INPUTS).cast(pl.Float32)).with_columns(pl.col('p2').fill_null(0.))
    if any(data[c].null_count() or not data[c].is_finite().all() for c in PRESENCE_INPUTS):
        raise ValueError('Invalid presence-model inputs')
    aggregates = [pl.len().cast(pl.Float32).log1p().alias('candidate_log_count'),
        (pl.col('ce_logit') > 0).mean().cast(pl.Float32).alias('ce_positive_fraction'),
        (pl.col('addr_len_r') == 0).mean().cast(pl.Float32).alias('missing_address_fraction')]
    for c in PRESENCE_INPUTS:
        aggregates += [pl.col(c).max().cast(pl.Float32).alias(c + '_max'),
                       pl.col(c).mean().cast(pl.Float32).alias(c + '_mean')]
    ranked = data.group_by('sidx').agg(aggregates)
    # Only within-business context: adding unrelated businesses cannot change it.
    top = data.group_by('sidx').agg(pl.col('ce_logit').sort(descending=True).head(2).alias('top'))
    top = top.select('sidx', ce_top_gap=(pl.col('top').list.first()
        - pl.col('top').list.get(1, null_on_oob=True).fill_null(pl.col('top').list.first())).cast(pl.Float32))
    return (anchors.select('sidx').join(ranked, on='sidx', how='left', validate='1:1')
            .join(top, on='sidx', how='left', validate='1:1')
            .with_columns(pl.exclude('sidx').fill_null(0.))
            .sort('sidx').with_columns(fold_expr()))


def fit_final(args):
    data = frame_for(args, 'train', [6, 7, 3])
    train = data.filter(pl.col('fold').is_in([6, 7]))
    valid = data.filter((pl.col('fold') == 3) & (r10.half() == 0))
    shift_path = args.work / 'shift_check.json'
    shift = json.loads(shift_path.read_text()) if shift_path.exists() else {}
    columns = model_features(data, 'enhanced' if shift.get('allow_enhanced', False) else 'baseline')
    info = {'version': VERSION, 'families': {}, 'presence': {}}
    for family, features in columns.items():
        info['families'][family] = {'features': features, 'variants': {}}
        for kind, params in r10.VARIANTS.items():
            model = fit(r10.weights(train) if kind == 'business' else train, features,
                        r10.weights(valid) if kind == 'business' else valid, args.rounds,
                        {**PARAMS, **params, 'device': args.device, 'nthread': args.threads,
                         'max_cached_hist_node': 1024})
            model.save_model(args.work / 'models' / f'r16_{family}_{kind}.json')
            info['families'][family]['variants'][kind] = {'best_iteration': model.best_iteration}
            del model
            gc.collect()
    country = r10.anchors(args).filter(fold_expr().is_in([6, 7, 3]))
    business = business_frame(data, country)
    # Use full organizer truth, not the surviving pair labels: an unretrieved
    # true match must not turn a positive business into a training singleton.
    positive = r10.truth(args, [6, 7, 3]).select('sidx').unique().with_columns(label=pl.lit(1, pl.Int8))
    business = business.join(positive, on='sidx', how='left').with_columns(pl.col('label').fill_null(0))
    tr = business.filter(pl.col('fold').is_in([6, 7]))
    va = business.filter((pl.col('fold') == 3) & (r10.half() == 0))
    features = [c for c in business.columns if c not in ('sidx', 'fold', 'label')]
    path = args.work / 'models/r16_presence.json'
    if tr.is_empty() or va.is_empty():
        raise ValueError('Presence model needs training and validation businesses')
    if tr['label'].n_unique() < 2:
        constant = float(tr['label'].mean())
        save_json(path, {'constant': constant})
        info['presence'] = {'features': features, 'constant': constant}
    else:
        model = fit(tr, features, va, args.rounds,
            {**PARAMS, 'max_depth': 4, 'eta': .04, 'reg_lambda': 10., 'seed': 1616,
             'device': args.device, 'nthread': args.threads, 'max_cached_hist_node': 256})
        model.save_model(path)
        info['presence'] = {'features': features, 'best_iteration': model.best_iteration}
    info['protocol'] = {'train_folds': [6, 7], 'early_stopping': '3A',
                        'presence_uses_fitted_pair_predictions': False}
    save_json(args.work / 'r16_models.json', info)


def predict_pair(args, frame, family, kind='mean'):
    meta = json.loads((args.work / 'r16_models.json').read_text())['families'][family]
    kinds = list(r10.VARIANTS) if kind == 'mean' else [kind]
    values = np.zeros(len(frame), np.float32)
    for name in kinds:
        model = xgb.Booster(model_file=str(args.work / 'models' / f'r16_{family}_{name}.json'))
        model.set_param({'device': args.device, 'nthread': args.threads})
        values += predict_frame(model, frame, meta['features'], args.batch_rows) / len(kinds)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite pair predictions')
    return frame.select(KEYS).with_columns(score=pl.Series(values))


def predict_presence(args, frame, country):
    meta = json.loads((args.work / 'r16_models.json').read_text())['presence']
    business = business_frame(frame, country)
    if 'constant' in meta:
        values = np.full(len(business), meta['constant'], np.float32)
    else:
        model = xgb.Booster(model_file=str(args.work / 'models/r16_presence.json'))
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
        raise ValueError('Missing/non-finite business probabilities')
    # Apply before target ownership: a suppressed business must not steal a
    # target from another eligible business. Threshold fitting does the same.
    return joined.filter(pl.col('presence') >= cutoff).select(KEYS + ['score'])


def calibrated(scores, presence, cutoff, target, country, kind, weight):
    subset = scores.join(country.select('sidx'), on='sidx', how='semi')
    allowed = eligible(subset, presence, cutoff)
    _, threshold = tune_threshold(allowed, target, country['sidx'], 'score')
    cutoffs = tune_country_thresholds(allowed, target, country, threshold, 'score')
    chosen = decide_country(allowed, threshold, country, cutoffs, 'score')
    return {'model': kind, 'weight': weight, 'presence_cutoff': cutoff,
            'threshold': threshold, 'country_thresholds': cutoffs,
            'fold3A': macro_f05(chosen, target, country['sidx'])['macro_f05']}


def decide(scores, presence, proposal, country):
    return decide_country(eligible(scores, presence, proposal['presence_cutoff']),
                          proposal['threshold'], country, proposal['country_thresholds'], 'score')


def select(args):
    country = r10.anchors(args).filter(fold_expr() == 3)
    a, b = country.filter(r10.half() == 0), country.filter(r10.half() == 1)
    target = r10.truth(args, [3])
    frame = frame_for(args, 'train', [3])
    base = predict_pair(args, frame, 'reference')
    presence = predict_presence(args, frame, country)
    # R12's measured winning architecture: mean of pair and business ensembles.
    # Recalibrate on 3A; its historical numeric threshold is never copied.
    reference = calibrated(base, presence, 0., target, a, 'reference', 0.)
    best, chosen, trials = reference, base, []
    for kind in ('reference', 'pair', 'business', 'mean'):
        current = base if kind == 'reference' else predict_pair(args, frame, 'enhanced', kind)
        for weight in ((0.,) if kind == 'reference' else (.5, 1.)):
            scores = base if kind == 'reference' else r10.blend(current, base, weight)
            for cutoff in (0., .1, .25, .5):
                trial = calibrated(scores, presence, cutoff, target, a, kind, weight)
                trials.append(trial)
                if trial['fold3A'] > best['fold3A'] + 1e-12:
                    best, chosen = trial, scores
    chosen_b = decide(chosen.join(b.select('sidx'), on='sidx', how='semi'), presence, best, b)
    base_b = decide(base.join(b.select('sidx'), on='sidx', how='semi'), presence, reference, b)
    comparison = r10.gate(chosen_b, base_b, target, b, args.minimum_gain)
    # Tighter per-country regression guard than the inherited graph gate.
    comparison['country_max_drop'] = .0005
    comparison['country_ok'] = all(comparison['candidate_by_country'][c]['macro_f05'] >=
        comparison['baseline_by_country'][c]['macro_f05'] - .0005 for c in comparison['baseline_by_country'])
    comparison['passed'] = bool(comparison['gain'] >= args.minimum_gain and
        comparison['approx_one_sided_95_lower'] > 0 and comparison['country_ok'])
    result = {'version': VERSION, 'selected': 'r16' if comparison['passed'] else 'reference',
        'reference': reference, 'proposal': best, 'gate': comparison, 'trials_on_3A': trials,
        'protocol_note': 'Final layer trains on 6/7, selects on 3A, gates once on 3B; earlier layers use fold 3. '
                         'Fold 4 is report-only here but inspected previously. France has no labeled evaluation.'}
    save_json(args.work / 'selection.json', result)
    print(json.dumps(result, indent=2), flush=True)


def chosen_scores(args, split, fold=None):
    selection = json.loads((args.work / 'selection.json').read_text())
    country = r10.anchors(args, split)
    if fold is not None:
        country = country.filter(fold_expr() == fold)
    frame = frame_for(args, split, [fold] if fold is not None else None)
    base = predict_pair(args, frame, 'reference')
    proposal = selection['proposal'] if selection['selected'] == 'r16' else selection['reference']
    current = base if proposal['model'] == 'reference' else r10.blend(
        predict_pair(args, frame, 'enhanced', proposal['model']), base, proposal['weight'])
    presence = predict_presence(args, frame, country)
    return selection, current, base, presence, country, proposal


def evaluate(args):
    selection, scores, base, presence, country, proposal = chosen_scores(args, 'train', 4)
    target = r10.truth(args, [4])
    chosen = decide(scores, presence, proposal, country)
    baseline = decide(base, presence, selection['reference'], country)
    metric = macro_f05(chosen, target, country['sidx'])
    from .r13_diagnostics import error_report
    singleton_ids = country.join(target.select('sidx').unique(), on='sidx', how='anti')['sidx']
    false_singletons = chosen.select('sidx').unique().filter(pl.col('sidx').is_in(singleton_ids))
    report = {'version': VERSION, 'selected': selection['selected'], 'local_fold4': metric,
        'reference_fold4': macro_f05(baseline, target, country['sidx']),
        'candidate_oracle_fold4': macro_f05(scores.join(target, on=KEYS), target, country['sidx']),
        'by_country': by_country(chosen, target, country), 'target_local': args.target_local,
        'target_met_locally': metric['macro_f05'] >= args.target_local,
        'target_leaderboard': args.target_leaderboard, 'amazon_score': None, 'target_met_on_leaderboard': None,
        'singletons': {'businesses': len(singleton_ids), 'false_matched_businesses': len(false_singletons)},
        'errors': error_report(args.work, scores, chosen, target), 'selection': selection,
        'limitations': ['No measured France or website score.', selection['protocol_note'],
            'Reference is rebuilt with the R12 sample recipe and L4 BF16; it is not the saved historical R12 artifact.']}
    save_json(args.work / 'metrics.json', report)
    print(json.dumps(report, indent=2), flush=True)


def inference(args):
    from .predict import write_lists
    _, scores, _, presence, country, proposal = chosen_scores(args, 'test')
    matches = decide(scores, presence, proposal, country)
    args.output.mkdir(parents=True, exist_ok=True)
    s1 = pl.read_parquet(args.work / 'norm/test_source1.parquet', columns=['idx', 'entity_id'])
    targets = pl.concat([pl.read_parquet(args.work / f'norm/test_source{i}.parquet', columns=['entity_id'])
                         for i in (2, 3)])['entity_id']
    # Every scored candidate stays in the candidate output, including pairs
    # rejected by the business classifier. All Source 1 rows are written.
    write_lists(args.output / 'candidate_pairs.tsv', s1, scores, targets, 'candidate_entity_ids')
    write_lists(args.output / 'matching_results.tsv', s1, matches, targets, 'matched_entity_ids')
    scores.write_parquet(args.work / 'r16_test_predictions.parquet')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['fit', 'select', 'evaluate', 'inference'])
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--dataset', type=Path, default=Path('student_resource/dataset'))
    p.add_argument('--output', type=Path, default=Path('output/r16'))
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--threads', type=int, default=30)
    p.add_argument('--rounds', type=int, default=1500)
    p.add_argument('--batch-rows', type=int, default=100000)
    p.add_argument('--minimum-gain', type=float, default=.0002)
    p.add_argument('--target-local', type=float, default=.99)
    p.add_argument('--target-leaderboard', type=float, default=.99)
    args = p.parse_args()
    if min(args.threads, args.rounds, args.batch_rows) < 1 or not 0 <= args.minimum_gain <= 1 or not 0 < args.target_local <= 1 or not 0 < args.target_leaderboard <= 1:
        p.error('Invalid runtime counts, gain or reporting target')
    {'fit': fit_final, 'select': select, 'evaluate': evaluate, 'inference': inference}[args.stage](args)


if __name__ == '__main__':
    main()
