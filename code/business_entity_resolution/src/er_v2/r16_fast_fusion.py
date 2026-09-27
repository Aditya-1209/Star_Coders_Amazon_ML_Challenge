"""Fast R16: fit CE donor fusion on 3A; preserve the actual R12 fallback."""
from __future__ import annotations

import argparse
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
from .predict import write_lists
from .r10_retrieval import save_json
from .run_block import load_split

VERSION = 'r16-fast-ce-fusion-1'
KEYS = ['sidx', 'tidx']
WEIGHTS = (.1, .25, .5, 1.)
MINIMUM_GAIN = .0005


def parent(args):
    return SimpleNamespace(**{**vars(args), 'work': args.base_work})


def maps(args, split):
    anchors, targets = load_split(args.base_work, split, columns=['idx', 'entity_id', 'country'])
    anchors = anchors.rename({'idx': 'sidx'})
    targets = targets.rename({'idx': 'tidx'})
    for frame, key in ((anchors, 'sidx'), (targets, 'tidx')):
        if frame['entity_id'].null_count() or frame['entity_id'].n_unique() != len(frame):
            raise ValueError('Organizer entity IDs are missing or ambiguous across sources')
        if not np.array_equal(frame[key].to_numpy(), np.arange(len(frame))):
            raise ValueError('Parent normalized indices must retain original row order')
    return anchors, targets


def checked_scores(path, anchors, targets, fold=None, *, baseline=False, score_kind='probability', score_column='score', donor_ids=None):
    """Always align on entity IDs; R14 numeric indices are deliberately ignored."""
    columns = pl.scan_parquet(path).collect_schema().names()
    if {'sid', 'tid'} <= set(columns):
        aliases = {'sid': 'sid', 'tid': 'tid'}
    elif {'source1_entity_id', 'target_entity_id'} <= set(columns):
        aliases = {'source1_entity_id': 'sid', 'target_entity_id': 'tid'}
    elif donor_ids is not None and set(KEYS) <= set(columns):
        aliases = None
    else:
        raise ValueError(f'{path}: entity-ID columns or the original R14 norm mappings are required')
    if aliases is None:
        data = pl.read_parquet(path, columns=[*KEYS, score_column]).rename({score_column: 'score'})
        source_ids, target_ids = donor_ids
        if sum(data.null_count().row(0)) or any(not data[k].dtype.is_integer() for k in KEYS):
            raise ValueError('R14 local indices must be non-null integers')
        if len(data) and (data['sidx'].min() < 0 or data['tidx'].min() < 0 or data['sidx'].max() >= len(source_ids) or data['tidx'].max() >= len(target_ids)):
            raise ValueError('R14 index outside its ORIGINAL norm mapping')
        data = data.select(sid=source_ids.gather(data['sidx']), tid=target_ids.gather(data['tidx']), score='score')
    else:
        data = pl.read_parquet(path, columns=[*aliases, score_column, *KEYS] if baseline else [*aliases, score_column]).rename({**aliases, score_column: 'score'})
    if sum(data.null_count().row(0)) or not data['score'].is_finite().all():
        raise ValueError(f'{path}: null or non-finite score/ID')
    if data.select('sid', 'tid').n_unique() != len(data):
        raise ValueError(f'{path}: duplicate entity pair')
    if score_kind == 'logit':
        data = data.with_columns(score=pl.Series(1 / (1 + np.exp(-np.clip(data['score'].to_numpy(), -80, 80)))))
    if not data['score'].is_between(0, 1).all():
        raise ValueError(f'{path}: expected probability scores in [0,1]')
    if baseline:
        if len(data) and (data['sidx'].min() < 0 or data['tidx'].min() < 0 or data['sidx'].max() >= len(anchors) or data['tidx'].max() >= len(targets)):
            raise ValueError('R12 index outside parent mapping')
        if not anchors['entity_id'].gather(data['sidx']).eq_missing(data['sid']).all() or not targets['entity_id'].gather(data['tidx']).eq_missing(data['tid']).all():
            raise ValueError('R12 exported IDs differ from the completed parent mapping')
    else:
        data = (data.join(anchors.select('sidx', sid=pl.col('entity_id')), on='sid', how='left', validate='m:1')
                    .join(targets.select('tidx', tid=pl.col('entity_id')), on='tid', how='left', validate='m:1'))
        if data['sidx'].null_count() or data['tidx'].null_count():
            raise ValueError('R14 includes entity IDs outside the organizer split')
    if fold is not None and not data.select(fold_expr())['fold'].eq(fold).all():
        raise ValueError('R14/R12 rows do not match the R12 row-index fold; do not rehash R14 row indices')
    if not anchors['country'].gather(data['sidx']).eq_missing(targets['country'].gather(data['tidx'])).all():
        raise ValueError('Cross-country candidate pair')
    return data.select(pl.col(KEYS).cast(pl.UInt32), pl.col('score').cast(pl.Float32))


COMPONENTS = ('graph', 'ce_a', 'ce_a_swapped', 'ce_b')


def read_donor(args, split, fold, anchors, targets):
    """Read PRE-fusion predictions, never the cross-fitted R14 final scores."""
    metadata = json.loads(args.r14_provenance.read_text())
    key = 'test' if split == 'test' else f'fold{fold}'
    donor_ids = None
    if metadata.get('norm_dir'):
        norm = args.r14_provenance.parent / metadata['norm_dir']
        source_ids = pl.read_parquet(norm / f'{split}_source1.parquet', columns=['entity_id'])['entity_id']
        target_ids = pl.concat([pl.read_parquet(norm / f'{split}_source{i}.parquet', columns=['entity_id']) for i in (2, 3)])['entity_id']
        if any(x.null_count() or x.n_unique() != len(x) for x in (source_ids, target_ids)):
            raise ValueError('R14 mapping entity IDs are missing or ambiguous')
        donor_ids = source_ids, target_ids
    outputs = []
    for component in COMPONENTS:
        spec = metadata['components'][component]
        path = args.r14_provenance.parent / spec['files'][key]
        rows = checked_scores(path, anchors, targets, fold, score_kind=spec['score_kind'], score_column=spec['score_column'], donor_ids=donor_ids)
        outputs.append(rows.rename({'score': component}))
    result = outputs[0]
    for component, rows in zip(COMPONENTS[1:], outputs[1:]):
        # CE is allowed only on the recorded graph candidate band. Missing
        # components remain null, never confidence zero or a negative label.
        if rows.join(result.select(KEYS), on=KEYS, how='anti').height:
            raise ValueError('CE component contains pairs outside the donor graph candidates')
        result = result.join(rows, on=KEYS, how='left', validate='1:1')
    return result


def scores_for(args, split, fold=None, *, include_r14=True):
    anchors, targets = maps(args, split)
    stem = 'test' if split == 'test' else f'fold{fold}'
    base = checked_scores(args.r12_export / f'r12_{stem}.parquet', anchors, targets, fold, baseline=True)
    other = read_donor(args, split, fold, anchors, targets) if include_r14 else None
    return anchors, targets, base, other


def feature_frame(base, other):
    union = base.rename({'score': 'r12'}).join(other, on=KEYS, how='full', coalesce=True, validate='1:1')
    raw = ['r12', *COMPONENTS]
    union = union.with_columns(*[pl.col(c).is_not_null().cast(pl.Float32).alias(c + '_available') for c in raw])
    union = union.with_columns(
        ce_mean=pl.mean_horizontal('ce_a', 'ce_a_swapped', 'ce_b'),
        ce_min=pl.min_horizontal('ce_a', 'ce_a_swapped', 'ce_b'),
        ce_max=pl.max_horizontal('ce_a', 'ce_a_swapped', 'ce_b'),
        graph_delta=pl.col('graph') - pl.col('r12'),
        swapped_delta=pl.col('ce_a_swapped') - pl.col('ce_a'),
        ce_b_delta=pl.col('ce_b') - pl.col('ce_a'))
    union = union.with_columns(
        ce_spread=pl.col('ce_max') - pl.col('ce_min'),
        ce_gap_s=pl.col('ce_mean').max().over('sidx') - pl.col('ce_mean'),
        ce_rank_s=pl.col('ce_mean').rank('min', descending=True).over('sidx').cast(pl.Float32))
    return union.sort(KEYS)


def feature_names():
    return ['r12', *COMPONENTS, *[c + '_available' for c in ('r12', *COMPONENTS)],
            'ce_mean', 'ce_min', 'ce_max', 'graph_delta', 'swapped_delta', 'ce_b_delta',
            'ce_spread', 'ce_gap_s', 'ce_rank_s']


def matrix(frame, truth=None):
    values = frame.select(feature_names()).to_numpy().astype(np.float32, copy=False)
    if truth is None:
        return xgb.DMatrix(values, feature_names=feature_names())
    labeled = frame.select(KEYS).join(truth.with_columns(label=pl.lit(1.)), on=KEYS, how='left', maintain_order='left', validate='m:1')
    labels = labeled['label'].fill_null(0.).to_numpy()
    weights = frame.select(w=(len(frame) / frame['sidx'].n_unique() / pl.len().over('sidx')))['w'].to_numpy()
    return xgb.DMatrix(values, label=labels, weight=weights, feature_names=feature_names())


def fit_fusion(frame_a, truth_a, args):
    """Optimizer sees 3A only; internal 3A split selects boosting rounds."""
    if not frame_a.select((fold_expr() == 3) & (r10.half() == 0)).to_series().all():
        raise ValueError('Fusion optimizer must receive only frozen R12 3A businesses')
    if not truth_a.select((fold_expr() == 3) & (r10.half() == 0)).to_series().all():
        raise ValueError('Fusion truth contains gate/holdout businesses')
    internal = (pl.col('sidx').hash(seed=1616) % 5) == 0
    train, valid = frame_a.filter(~internal), frame_a.filter(internal)
    if min(train['sidx'].n_unique(), valid['sidx'].n_unique()) < 2:
        raise ValueError('Too few 3A businesses for fusion training/early stopping')
    params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'tree_method': 'hist',
              'device': args.device, 'nthread': args.threads, 'seed': 1616, 'max_depth': 4,
              'eta': .04, 'min_child_weight': 20, 'reg_lambda': 20., 'subsample': 1.,
              'colsample_bytree': 1., 'max_bin': 128}
    initial = xgb.train(params, matrix(train, truth_a), num_boost_round=args.rounds,
                        evals=[(matrix(valid, truth_a), '3A_internal')], early_stopping_rounds=30, verbose_eval=False)
    rounds = initial.best_iteration + 1
    model = xgb.train(params, matrix(frame_a, truth_a), num_boost_round=rounds)
    if args.device == 'cuda' and json.loads(model.save_config())['learner']['generic_param']['device'] == 'cpu':
        raise RuntimeError('XGBoost fell back to CPU despite requested CUDA')
    model.save_model(args.work / 'fusion.json')
    return model, {'rounds': rounds, 'features': feature_names(), 'fit_partition': '3A',
                   'early_stopping_partition': 'hash(sidx,1616)%5==0 within 3A', 'parameters': params}


def fusion_scores(frame, model):
    pieces = []
    for part in frame.iter_slices(500000):
        pred = model.predict(matrix(part))
        # R12-only candidates retain their original score, even at weight 1.
        pieces.append(part.select(*KEYS, 'r12', 'graph').with_columns(fusion=pl.Series(pred)))
    return pl.concat(pieces)


def mixture(frame, weight):
    if not 0 <= weight <= 1:
        raise ValueError('Invalid ensemble weight')
    if weight == 0:
        return frame.filter(pl.col('r12').is_not_null()).select(*KEYS, score=pl.col('r12'))
    return frame.select(*KEYS, score=pl.when(pl.col('graph').is_null()).then(pl.col('r12'))
        .when(pl.col('r12').is_null()).then(pl.col('fusion'))
        .otherwise((1 - weight) * pl.col('r12') + weight * pl.col('fusion')).cast(pl.Float32))


def decide(scores, proposal, country):
    return decide_country(scores, proposal['threshold'], country, proposal['country_thresholds'], 'score')


def select(args):
    anchors, targets, base, other = scores_for(args, 'train', 3)
    country = anchors.select('sidx', 'country').filter(fold_expr() == 3)
    del anchors, targets
    a, b = country.filter(r10.half() == 0), country.filter(r10.half() == 1)
    truth = r10.truth(parent(args), [3])
    truth_a = truth.join(a.select('sidx'), on='sidx', how='semi')
    old = json.loads((args.base_work / 'selection.json').read_text())
    base_a = base.join(a.select('sidx'), on='sidx', how='semi')
    base_b = base.join(b.select('sidx'), on='sidx', how='semi')
    baseline_a = r10.final_decision(parent(args), old, base_a, a)
    baseline_b = r10.final_decision(parent(args), old, base_b, b)
    baseline_value = macro_f05(baseline_a, truth_a, a['sidx'])['macro_f05']
    if old['selected'] == 'r10' and abs(baseline_value - old['proposal']['fold3A']) > 1e-9:
        raise ValueError('R12 archive does not reproduce the frozen 3A score')
    expected_b = old['gate']['candidate_by_country' if old['selected'] == 'r10' else 'baseline_by_country']
    actual_b = by_country(baseline_b, truth, b)
    if set(actual_b) != set(expected_b) or any(abs(actual_b[c][k] - expected_b[c][k]) > 1e-9 for c in actual_b for k in actual_b[c]):
        raise ValueError('R12 archive does not reproduce the frozen 3B reference')
    frame = feature_frame(base, other)
    frame_a = frame.join(a.select('sidx'), on='sidx', how='semi')
    model, fit_metadata = fit_fusion(frame_a, truth_a, args)
    fused_a = fusion_scores(frame_a, model)
    best, best_value, trials = None, baseline_value, []
    for weight in WEIGHTS:
        scores = mixture(fused_a, weight)
        _, threshold = tune_threshold(scores, truth_a, a['sidx'], 'score')
        cutoffs = tune_country_thresholds(scores, truth_a, a, threshold, 'score')
        proposal = {'fusion_weight': weight, 'threshold': threshold, 'country_thresholds': cutoffs}
        value = macro_f05(decide(scores, proposal, a), truth_a, a['sidx'])['macro_f05']
        trials.append({**proposal, 'fold3A': value})
        if value > best_value + 1e-12:
            best, best_value = trials[-1], value
    proposed = baseline_b
    if best is not None:
        fused_b = fusion_scores(frame.join(b.select('sidx'), on='sidx', how='semi'), model)
        proposed = decide(mixture(fused_b, best['fusion_weight']), best, b)
    gate = r10.gate(proposed, baseline_b, truth, b, MINIMUM_GAIN)
    result = {'version': VERSION, 'selected': 'r16_fast_ce_fusion' if best is not None and gate['passed'] else 'r12',
        'proposal': best, 'fusion_fit': fit_metadata, 'gate': gate, 'trials_on_3A': trials,
        'baseline_3A': baseline_value, 'base_selection': old,
        'r14_provenance': json.loads(args.r14_provenance.read_text()),
        'missing_scores': 'Union; R12-only stays R12, donor-only uses fusion, shared pairs blend; weight zero is exact R12.',
        'protocol_note': 'Fit fusion, early-stop and select on 3A; one unchanged .0005/positive lower-bound/.002 country gate on 3B. '
                         'Inherited layers and earlier experiments used these partitions. Fold4 is report-only.'}
    save_json(args.work / 'selection.json', result)
    print(json.dumps(result, indent=2), flush=True)


def chosen(args, split, fold=None):
    selection = json.loads((args.work / 'selection.json').read_text())
    anchors, targets, base, other = scores_for(args, split, fold, include_r14=selection['selected'] != 'r12')
    scores = base
    if selection['selected'] != 'r12':
        model = xgb.Booster()
        model.load_model(args.work / 'fusion.json')
        model.set_param({'device': args.device, 'nthread': args.threads})
        fused = fusion_scores(feature_frame(base, other), model)
        scores = mixture(fused, selection['proposal']['fusion_weight'])
    return selection, anchors, targets, base, scores


def final_decision(args, selection, scores, country):
    if selection['selected'] == 'r12':
        return r10.final_decision(parent(args), selection['base_selection'], scores, country)
    return decide(scores, selection['proposal'], country)


def evaluate(args):
    selection, anchors, targets, base, scores = chosen(args, 'train', 4)
    country = anchors.select('sidx', 'country').filter(fold_expr() == 4)
    truth = r10.truth(parent(args), [4])
    baseline = r10.final_decision(parent(args), selection['base_selection'], base, country)
    old = json.loads((args.base_work / 'metrics.json').read_text())['local_fold4']
    measured = macro_f05(baseline, truth, country['sidx'])
    if any(abs(measured[key] - old[key]) > 1e-9 for key in measured):
        raise ValueError('R12 archive does not reproduce frozen fold4 results')
    pred = final_decision(args, selection, scores, country)
    save_json(args.work / 'metrics.json', {'version': VERSION, 'selected': selection['selected'],
        'local_fold4': macro_f05(pred, truth, country['sidx']), 'reference_fold4': measured,
        'by_country': by_country(pred, truth, country), 'selection': selection, 'amazon_score': None,
        'limitation': 'This is a previously examined labeled holdout; no measured France or website gain.'})


def inference(args):
    selection, anchors, targets, base, scores = chosen(args, 'test')
    pred = final_decision(args, selection, scores, anchors.select('sidx', 'country'))
    args.output.mkdir(parents=True, exist_ok=True)
    s1 = anchors.select(idx=pl.col('sidx'), entity_id=pl.col('entity_id'))
    for filename, rows, column in [('matching_results.tsv', pred, 'matched_entity_ids'),
                                  ('candidate_pairs.tsv', scores, 'candidate_entity_ids')]:
        write_lists(args.output / filename, s1, rows, targets['entity_id'], column)
    if selection['selected'] == 'r12':
        import hashlib
        old = json.loads((args.base_work / 'result.json').read_text())['output_sha256']
        for name in ('matching_results.tsv', 'candidate_pairs.tsv'):
            h = hashlib.sha256()
            with (args.output / name).open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024**2), b''): h.update(block)
            if h.hexdigest() != old[name]:
                raise ValueError('Fallback must be byte-identical to the original R12 outputs: ' + name)
    save_json(args.work / 'inference.json', {'selected': selection['selected'], 'candidates': len(scores),
        'matches': len(pred), 'source1_rows': len(anchors), 'exact_r12_fallback': selection['selected'] == 'r12'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['select', 'evaluate', 'inference'])
    for name in ('base-work', 'r12-export', 'r14-provenance', 'dataset', 'work', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--threads', type=int, default=24)
    parser.add_argument('--rounds', type=int, default=400)
    args = parser.parse_args()
    {'select': select, 'evaluate': evaluate, 'inference': inference}[args.stage](args)


if __name__ == '__main__':
    main()
