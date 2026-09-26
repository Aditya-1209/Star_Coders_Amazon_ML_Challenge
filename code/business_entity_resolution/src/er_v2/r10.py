"""R10 CE + graph stacker; tune on 3A, gate once on 3B, report on fold 4."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import polars as pl
import xgboost as xgb
from .decision import decide_country, tune_threshold, tune_country_thresholds
from .graph import hop_keep
from .metrics import macro_f05, by_country
from .r10_retrieval import save_json
from .run_block import load_split
from .run_features import ground_truth_pairs
from .stage3 import feature_cols
from .train import fit, PARAMS, fold_expr, predict_frame

VERSION = 'r10-ce-ann-1'
VARIANTS = {'pair': {'max_depth': 8, 'eta': .04, 'reg_lambda': 5., 'seed': 1010},
            'business': {'max_depth': 7, 'eta': .04, 'reg_lambda': 8., 'seed': 2010}}


def half():
    return (pl.col('sidx').hash(seed=1010) % 2).cast(pl.Int8)


def anchors(args, split='train'):
    return pl.read_parquet(args.work / 'norm' / f'{split}_source1.parquet', columns=['idx', 'country']).select(
        sidx=pl.col('idx').cast(pl.UInt32), country='country')


def truth(args, folds):
    s1, tg = load_split(args.work, 'train', columns=['idx', 'entity_id', 'country'])
    s1 = s1.filter((pl.col('idx').hash(seed=11) % 10).is_in(folds))
    return ground_truth_pairs(args.dataset, s1, tg)


def attach_ce(frame, scores):
    keys = ['sidx', 'tidx']
    if scores.select(keys).n_unique() != len(scores):
        raise ValueError('Duplicate cross-encoder pair keys')
    out = frame.join(scores, on=keys, how='left', validate='1:1')
    if out['ce_logit'].null_count() or not np.isfinite(out['ce_logit'].to_numpy()).all():
        raise ValueError('Missing/non-finite cross-encoder predictions; rerun score stage')
    # Competition features use scores, never labels or corpus size.
    out = out.sort('sidx', 'tidx')
    return out.with_columns(
        ce_rank_s=pl.col('ce_logit').rank('ordinal', descending=True).over('sidx').cast(pl.UInt16),
        ce_gap_s=(pl.col('ce_logit').max().over('sidx') - pl.col('ce_logit')).cast(pl.Float32),
        ce_rank_t=pl.col('ce_logit').rank('ordinal', descending=True).over('tidx').cast(pl.UInt16),
        ce_gap_t=(pl.col('ce_logit').max().over('tidx') - pl.col('ce_logit')).cast(pl.Float32))


def frame_for(args, split, folds=None):
    source = pl.scan_parquet(args.work / f'stage3_{split}.parquet').filter(hop_keep())
    if folds is not None:
        source = source.filter(pl.col('fold').is_in(folds))
    scores = pl.scan_parquet(args.work / f'ce_{split}.parquet')
    if folds is not None:
        scores = scores.filter(fold_expr().is_in(folds))
    return attach_ce(source.collect(engine='streaming'), scores.collect(engine='streaming'))


def weights(frame):
    return frame.with_columns(w=(len(frame) / frame['sidx'].n_unique() / pl.len().over('sidx')).cast(pl.Float32))


def fit_final(args):
    data = frame_for(args, 'train', [6, 7, 3])
    train = data.filter(pl.col('fold').is_in([6, 7]))
    valid = data.filter((pl.col('fold') == 3) & (half() == 0))
    shift = json.loads((args.work / 'shift_check.json').read_text()) if (args.work / 'shift_check.json').exists() else {}
    features = feature_cols(data, 'enhanced' if shift.get('allow_enhanced', False) else 'baseline')
    del data
    info = {'version': VERSION, 'features': features, 'variants': {}}
    for name, parameters in VARIANTS.items():
        model = fit(weights(train) if name == 'business' else train, features,
                    weights(valid) if name == 'business' else valid, args.rounds,
                    {**PARAMS, **parameters, 'device': args.device, 'nthread': args.threads,
                     'max_cached_hist_node': 1024})
        model.save_model(args.work / 'models' / f'r10_{name}.json')
        info['variants'][name] = {'best_iteration': model.best_iteration, 'parameters': parameters}
        del model
        gc.collect()
    save_json(args.work / 'r10_models.json', info)


def predict(args, frame, choice):
    meta = json.loads((args.work / 'r10_models.json').read_text())
    names = list(VARIANTS) if choice == 'mean' else [choice]
    values = np.zeros(len(frame), np.float32)
    for name in names:
        model = xgb.Booster(model_file=str(args.work / 'models' / f'r10_{name}.json'))
        model.set_param({'device': args.device, 'nthread': args.threads})
        values += predict_frame(model, frame, meta['features'], args.batch_rows) / len(names)
    return frame.select('sidx', 'tidx').with_columns(score=pl.Series(values))


def blend(candidate, baseline, weight):
    """Identical graph candidates are required; missing evidence must not become zero."""
    key = ['sidx', 'tidx']
    out = candidate.join(baseline.select(*key, base='score'), on=key, how='full', coalesce=True, validate='1:1')
    if out['score'].null_count() or out['base'].null_count():
        raise ValueError('Baseline and R10 candidate sets differ')
    return out.select(*key, score=(weight * pl.col('score') + (1 - weight) * pl.col('base')))


def baseline_scores(args, split, fold=None):
    path = 'eval_preds_stage3.parquet' if split == 'train' else 'test_preds_stage3.parquet'
    frame = pl.scan_parquet(args.work / path)
    if fold is not None:
        frame = frame.filter(pl.col('fold') == fold)
    return frame.select('sidx', 'tidx', 'score').collect()


def baseline_decide(args, scores, country):
    meta = json.loads((args.work / 'models/stage3_metrics.json').read_text())
    return decide_country(scores, meta['threshold'], country, meta.get('country_thresholds', {}), 'score')


def per_business(pred, target, ids):
    a = pl.DataFrame({'sidx': ids}).unique()
    p = pred.select('sidx', 'tidx').unique().join(a, on='sidx', how='semi')
    t = target.select('sidx', 'tidx').unique().join(a, on='sidx', how='semi')
    counts = a.join(p.group_by('sidx').agg(np=pl.len()), on='sidx', how='left')
    counts = counts.join(t.group_by('sidx').agg(nt=pl.len()), on='sidx', how='left')
    counts = counts.join(p.join(t, on=['sidx', 'tidx']).group_by('sidx').agg(tp=pl.len()), on='sidx', how='left').fill_null(0)
    return counts.select('sidx', f=pl.when(pl.col('np') + pl.col('nt') == 0).then(1.)
        .otherwise(1.25 * pl.col('tp') / (pl.col('np') + .25 * pl.col('nt'))))


def gate(candidate, baseline, target, country, minimum_gain):
    if len(country) < 2:
        raise ValueError('Gate requires at least two businesses')
    a = per_business(candidate, target, country['sidx'])
    b = per_business(baseline, target, country['sidx']).rename({'f': 'base'})
    delta = a.join(b, on='sidx').select(d=pl.col('f') - pl.col('base'))['d']
    gain = float(delta.mean())
    lower = gain - 1.645 * float(delta.std(ddof=1)) / np.sqrt(len(delta))
    ca, cb = by_country(candidate, target, country), by_country(baseline, target, country)
    country_ok = all(ca[c]['macro_f05'] >= cb[c]['macro_f05'] - .002 for c in cb)
    return {'gain': gain, 'approx_one_sided_95_lower': float(lower), 'minimum_gain': minimum_gain,
            'country_ok': country_ok, 'passed': bool(gain >= minimum_gain and lower > 0 and country_ok),
            'candidate_by_country': ca, 'baseline_by_country': cb}


def select(args):
    country = anchors(args).filter(fold_expr() == 3)
    a, b = country.filter(half() == 0), country.filter(half() == 1)
    target = truth(args, [3])
    frame = frame_for(args, 'train', [3])
    base = baseline_scores(args, 'train', 3)
    trial, best, chosen = [], None, None
    for name in (*VARIANTS, 'mean'):
        current = predict(args, frame, name)
        for weight in (.25, .5, .75, 1.):
            scores = blend(current, base, weight)
            value, threshold = tune_threshold(scores, target, a['sidx'], 'score')
            # All country cutoffs are fitted on A, including a global fallback for France.
            cutoffs = tune_country_thresholds(scores, target, a, threshold, 'score')
            value = macro_f05(decide_country(scores.join(a.select('sidx'), on='sidx', how='semi'),
                threshold, a, cutoffs, 'score'), target, a['sidx'])['macro_f05']
            entry = {'model': name, 'weight': weight, 'threshold': threshold,
                     'country_thresholds': cutoffs, 'fold3A': value}
            trial.append(entry)
            if best is None or value > best['fold3A'] + 1e-12:
                best, chosen = entry, scores
    selected_b = decide_country(chosen.join(b.select('sidx'), on='sidx', how='semi'), best['threshold'], b,
                                best['country_thresholds'], 'score')
    base_b = baseline_decide(args, base.join(b.select('sidx'), on='sidx', how='semi'), b)
    comparison = gate(selected_b, base_b, target, b, args.minimum_gain)
    result = {'version': VERSION, 'selected': 'r10' if comparison['passed'] else 'reference',
        'proposal': best, 'gate': comparison, 'trials_on_3A': trial,
        'protocol_note': '3B is held out only for the new final layer; inherited earlier layers use fold 3. Fold 4 is report-only here but has been inspected in past experiments.'}
    save_json(args.work / 'selection.json', result)
    print(json.dumps(result, indent=2), flush=True)


def chosen_scores(args, split, fold=None):
    selection = json.loads((args.work / 'selection.json').read_text())
    base = baseline_scores(args, split, fold)
    if selection['selected'] == 'reference':
        return selection, base
    frame = frame_for(args, split, [fold] if fold is not None else None)
    proposal = selection['proposal']
    return selection, blend(predict(args, frame, proposal['model']), base, proposal['weight'])


def final_decision(args, selection, scores, country):
    if selection['selected'] == 'reference':
        return baseline_decide(args, scores, country)
    p = selection['proposal']
    return decide_country(scores, p['threshold'], country, p['country_thresholds'], 'score')


def evaluate(args):
    # The saved choice is loaded before holdout labels; nothing is reselected here.
    selection, scores = chosen_scores(args, 'train', 4)
    country = anchors(args).filter(fold_expr() == 4)
    chosen = final_decision(args, selection, scores, country)
    baseline = baseline_decide(args, baseline_scores(args, 'train', 4), country)
    target = truth(args, [4])
    metrics = macro_f05(chosen, target, country['sidx'])
    report = {'version': VERSION, 'selected': selection['selected'], 'local_fold4': metrics,
        'reference_fold4': macro_f05(baseline, target, country['sidx']),
        'candidate_oracle_fold4': macro_f05(scores.join(target, on=['sidx', 'tidx']), target, country['sidx']),
        'by_country': by_country(chosen, target, country), 'target_local': .975,
        'target_met_locally': metrics['macro_f05'] >= .975, 'amazon_score': None,
        'user_reported_r8': {'local': .956, 'online': .954}, 'selection': selection,
        'limitations': ['No measured France score: training has no France labels.',
            'Rebuilt R8-style reference uses the R10 candidates, encoder and rescue rule; it is not the old submitted artifact.',
            'Target 97.5 is an objective, not a guaranteed or measured leaderboard result.']}
    save_json(args.work / 'metrics.json', report)
    print(json.dumps(report, indent=2), flush=True)


def inference(args):
    from .predict import write_lists
    selection, scores = chosen_scores(args, 'test')
    country = anchors(args, 'test')
    matches = final_decision(args, selection, scores, country)
    args.output.mkdir(parents=True, exist_ok=True)
    s1 = pl.read_parquet(args.work / 'norm/test_source1.parquet', columns=['idx', 'entity_id'])
    targets = pl.concat([pl.read_parquet(args.work / f'norm/test_source{i}.parquet', columns=['entity_id']) for i in (2, 3)])['entity_id']
    write_lists(args.output / 'candidate_pairs.tsv', s1, scores, targets, 'candidate_entity_ids')
    write_lists(args.output / 'matching_results.tsv', s1, matches, targets, 'matched_entity_ids')
    scores.write_parquet(args.work / 'r10_test_predictions.parquet')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['fit', 'select', 'evaluate', 'inference'])
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--dataset', type=Path, default=Path('student_resource/dataset'))
    p.add_argument('--output', type=Path, default=Path('output/r10'))
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--threads', type=int, default=12)
    p.add_argument('--rounds', type=int, default=1400)
    p.add_argument('--batch-rows', type=int, default=50000)
    p.add_argument('--minimum-gain', type=float, default=.0005)
    args = p.parse_args()
    if min(args.threads, args.rounds, args.batch_rows) < 1 or args.minimum_gain < 0:
        p.error('Invalid runtime counts or gain')
    {'fit': fit_final, 'select': select, 'evaluate': evaluate, 'inference': inference}[args.stage](args)

if __name__ == '__main__':
    main()
