"""Cached France transfer experiment. No encoder/CE fitting or new retrieval.

Fits on 6/7, chooses on 3A, gates once on 3B, reports on 4. Only France test
decisions may change. The labelled-country comparison is a transfer proxy,
not France validation. A failed gate reproduces the frozen R12 submission.
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
from .decision import decide_country, tune_threshold
from .features import feature_names
from .folds import fold_expr
from .france_features import records, similarities, transfer_weights
from .graph import hop_keep
from .metrics import macro_f05, by_country
from .predict import write_lists
from .r10_retrieval import save_json
from .run_block import load_split
from .train import PARAMS, fit, predict_frame

VERSION = 'r12-france-transfer-1'
# Target ranks have different population density in graph training and test.
EXCLUDE = {'ce_rank_t', 'ce_gap_t', 'ncos_rank_t', 'ncos_gap_t'}
DOMAIN_FEATURES = ['core_ratio', 'core_tsort', 'addr_tsort', 'addr_both_present',
                   'nonlatin_l', 'nonlatin_r', 'is_s3', 'fr_name_ratio',
                   'fr_addr_ratio', 'fr_name_cont_min', 'fr_name_tail']


def parent(args):
    return SimpleNamespace(work=args.base_work, dataset=args.dataset, device=args.device,
                           threads=args.threads, batch_rows=50000)


def countries(args, split='train'):
    return r10.anchors(parent(args), split)


def baseline(args, split, fold=None):
    if split == 'test':
        return pl.read_parquet(args.base_work / 'r10_test_predictions.parquet').select('sidx', 'tidx', 'score')
    return r10.chosen_scores(parent(args), split, fold)[1]


def base_decision(args, scores, country):
    selection = json.loads((args.base_work / 'selection.json').read_text())
    return r10.final_decision(parent(args), selection, scores, country)


def check_keys(frame):
    if frame.select('sidx', 'tidx').n_unique() != len(frame):
        raise ValueError('Duplicate candidate keys')


def prepare(args):
    split = 'test' if args.split == 'test' else 'train'
    folds = [4] if args.split == 'holdout' else [6, 7, 3]
    source = pl.scan_parquet(args.base_work / f'stage3_{split}.parquet').filter(hop_keep())
    if split == 'train':
        source = source.filter(pl.col('fold').is_in(folds))
    else:
        fr = countries(args, 'test').filter(pl.col('country') == 'France').select('sidx')
        if fr.is_empty():
            raise ValueError('France is absent from the supplied test dataset')
        source = source.join(fr.lazy(), on='sidx', how='semi')
    frame = source.collect(engine='streaming')
    # Context is based on all existing CE scores, then restricted to this frame.
    scores = pl.read_parquet(args.base_work / f'ce_{split}.parquet')
    frame = r10.attach_ce(frame, scores)
    del scores
    check_keys(frame)
    s1, tg = load_split(args.base_work, split, 'France' if split == 'test' else None,
        ['idx', 'country', 'business_name', 'business_address'])
    left = records(s1.join(frame.select(idx='sidx').unique(), on='idx', how='semi'))
    right = records(tg.join(frame.select(idx='tidx').unique(), on='idx', how='semi'))
    del s1, tg
    extras = [similarities(part, left, right, args.threads) for part in frame.iter_slices(100000)]
    frame = frame.join(pl.concat(extras), on=['sidx', 'tidx'], how='left', validate='1:1')
    path = args.work / f'features_{args.split}.parquet'
    temporary = path.with_suffix('.partial.parquet')
    frame.write_parquet(temporary)
    temporary.replace(path)
    print(f'{args.split}: {len(frame):,} pairs; {frame["sidx"].n_unique():,} businesses', flush=True)


def fit_models(args):
    data = pl.read_parquet(args.work / 'features_train.parquet')
    train = r10.weights(data.filter(pl.col('fold').is_in([6, 7])))
    valid = r10.weights(data.filter((pl.col('fold') == 3) & (r10.half() == 0)))
    del data
    features = [c for c in feature_names(train) if c not in EXCLUDE]
    # Equal-prior, bounded domain sample. No France pseudo-labels, match scores,
    # fold-3/4 rows, entity IDs or countries enter the domain classifier.
    fr = pl.read_parquet(args.work / 'features_test.parquet', columns=DOMAIN_FEATURES)
    count = min(250000, len(fr), len(train))
    source = train.select(DOMAIN_FEATURES).sample(n=count, seed=121)
    target = fr.sample(n=count, seed=122)
    domain_data = pl.concat([source.with_columns(label=pl.lit(0)), target.with_columns(label=pl.lit(1))])
    domain = fit(domain_data, DOMAIN_FEATURES, None, 64,
        {**PARAMS, 'max_depth': 3, 'eta': .05, 'min_child_weight': 50, 'reg_lambda': 10.,
         'device': args.device, 'nthread': args.threads, 'seed': 123})
    domain.save_model(args.work / 'models/domain.json')
    del fr, source, target, domain_data
    ratio, diag = transfer_weights(predict_frame(domain, train, DOMAIN_FEATURES, 50000), train['w'])
    vratio, _ = transfer_weights(predict_frame(domain, valid, DOMAIN_FEATURES, 50000), valid['w'])
    del domain
    info = {'version': VERSION, 'features': features, 'train_folds': [6, 7],
            'early_stopping': '3A', 'domain': {'features': DOMAIN_FEATURES, 'sample_per_domain': count,
            'rounds': 64, 'weight_diagnostics': diag}, 'variants': {}}
    for name in ('balanced', 'transfer'):
        tr = train if name == 'balanced' else train.with_columns(w=pl.Series(ratio))
        va = valid if name == 'balanced' else valid.with_columns(w=pl.Series(vratio))
        model = fit(tr, features, va, args.rounds,
            {**PARAMS, 'max_depth': 7, 'eta': .05, 'reg_lambda': 8., 'min_child_weight': 10,
             'device': args.device, 'nthread': args.threads, 'max_cached_hist_node': 512, 'seed': 1212})
        model.save_model(args.work / f'models/{name}.json')
        info['variants'][name] = {'best_iteration': model.best_iteration}
        del model, tr, va
        gc.collect()
    save_json(args.work / 'models.json', info)


def proposed_scores(args, frame, variant):
    meta = json.loads((args.work / 'models.json').read_text())
    model = xgb.Booster(model_file=str(args.work / f'models/{variant}.json'))
    model.set_param({'device': args.device, 'nthread': args.threads})
    return frame.select('sidx', 'tidx').with_columns(
        score=pl.Series(predict_frame(model, frame, meta['features'], 50000)))


def select(args):
    c = countries(args).filter(fold_expr() == 3)
    a, b = c.filter(r10.half() == 0), c.filter(r10.half() == 1)
    target = r10.truth(parent(args), [3])
    base = baseline(args, 'train', 3)
    base_a = base_decision(args, base.join(a.select('sidx'), on='sidx', how='semi'), a)
    base_value = macro_f05(base_a, target, a['sidx'])['macro_f05']
    frame = pl.read_parquet(args.work / 'features_train.parquet').filter(pl.col('fold') == 3)
    best, chosen, trials = None, None, []
    for variant in ('balanced', 'transfer'):
        pred = proposed_scores(args, frame, variant)
        for weight in (.25, .5, 1.):
            scores = r10.blend(pred, base, weight)
            value, threshold = tune_threshold(scores, target, a['sidx'], 'score')
            entry = {'model': variant, 'weight': weight, 'threshold': threshold, 'fold3A': value}
            trials.append(entry)
            if best is None or value > best['fold3A'] + 1e-12:
                best, chosen = entry, scores
    # A global cutoff only: unseen France has no tuning labels.
    new_b = decide_country(chosen.join(b.select('sidx'), on='sidx', how='semi'), best['threshold'], b, {}, 'score')
    base_b = base_decision(args, base.join(b.select('sidx'), on='sidx', how='semi'), b)
    gate = r10.gate(new_b, base_b, target, b, .0005)
    passed = best['fold3A'] > base_value + 1e-12 and gate['passed']
    result = {'version': VERSION, 'selected': 'france_transfer' if passed else 'r12',
              'proposal': best, 'baseline_fold3A': base_value, 'gate': gate, 'trials_on_3A': trials,
              'scope': 'France test rows only; India/US use frozen R12',
              'limitation': '3B tests labelled-country transfer proxies, not France. Earlier models used fold 3; these partitions have been examined before.'}
    save_json(args.work / 'selection.json', result)
    print(json.dumps(result, indent=2), flush=True)


def evaluate(args):
    selection = json.loads((args.work / 'selection.json').read_text())  # Freeze before reading 4.
    c = countries(args).filter(fold_expr() == 4)
    base = baseline(args, 'train', 4)
    base_matches = base_decision(args, base, c)
    target = r10.truth(parent(args), [4])
    p = selection['proposal']
    frame = pl.read_parquet(args.work / 'features_holdout.parquet')
    scores = r10.blend(proposed_scores(args, frame, p['model']), base, p['weight'])
    candidate = decide_country(scores, p['threshold'], c, {}, 'score')
    result = {'version': VERSION, 'selected': selection['selected'],
        'local_fold4': macro_f05(base_matches, target, c['sidx']),
        'proposal_proxy_fold4': macro_f05(candidate, target, c['sidx']),
        'proposal_proxy_by_country': by_country(candidate, target, c),
        'baseline_by_country': by_country(base_matches, target, c),
        'france_score': None, 'amazon_score': None, 'selection': selection,
        'limitations': ['No labelled France holdout: proxy improvements cannot establish France accuracy.',
            'Local deployed decisions stay R12: the patch applies only to France, absent from training.',
            'Candidate set is inherited: this update cannot recover pairs missing from R12 retrieval.',
            'Density weighting assumes transferable conditional labels; it can fail under language shift.']}
    save_json(args.work / 'metrics.json', result)
    print(json.dumps(result, indent=2), flush=True)


def replace_france(base_matches, france_matches, country):
    fr = country.filter(pl.col('country') == 'France').select('sidx')
    if not france_matches.join(fr, on='sidx', how='anti').is_empty():
        raise ValueError('France update contains a non-France business')
    kept = base_matches.join(fr, on='sidx', how='anti').select('sidx', 'tidx')
    if not kept.join(france_matches, on='tidx').is_empty():
        raise ValueError('France update attempts to take a non-France target')
    return pl.concat([kept, france_matches.select('sidx', 'tidx')])


def inference(args):
    selection = json.loads((args.work / 'selection.json').read_text())
    country = countries(args, 'test')
    scores = baseline(args, 'test')
    check_keys(scores)
    matches = base_decision(args, scores, country)
    old_matches = matches.select('sidx', 'tidx')
    if selection['selected'] == 'france_transfer':
        fr = country.filter(pl.col('country') == 'France')
        frame = pl.read_parquet(args.work / 'features_test.parquet')
        p = selection['proposal']
        updated = r10.blend(proposed_scores(args, frame, p['model']),
                           scores.join(fr.select('sidx'), on='sidx', how='semi'), p['weight'])
        fr_matches = decide_country(updated, p['threshold'], fr, {}, 'score')
        matches = replace_france(old_matches, fr_matches, country)
        updated.write_parquet(args.work / 'france_predictions.parquet')
    else:
        scores.join(country.filter(pl.col('country') == 'France').select('sidx'), on='sidx', how='semi').write_parquet(
            args.work / 'france_predictions.parquet')
    other = country.filter(pl.col('country') != 'France').select('sidx')
    if not matches.select('sidx', 'tidx').join(other, on='sidx', how='semi').sort('sidx', 'tidx').equals(
            old_matches.join(other, on='sidx', how='semi').sort('sidx', 'tidx')):
        raise ValueError('Non-France decisions changed')
    args.output.mkdir(parents=True, exist_ok=True)
    s1, tg = load_split(args.base_work, 'test', columns=['idx', 'entity_id'])
    write_lists(args.output / 'candidate_pairs.tsv', s1, scores, tg['entity_id'], 'candidate_entity_ids')
    write_lists(args.output / 'matching_results.tsv', s1, matches, tg['entity_id'], 'matched_entity_ids')
    added = matches.select('sidx', 'tidx').join(old_matches, on=['sidx', 'tidx'], how='anti')
    removed = old_matches.join(matches.select('sidx', 'tidx'), on=['sidx', 'tidx'], how='anti')
    save_json(args.work / 'changes.json', {'selected': selection['selected'],
        'added_pairs': added.height, 'removed_pairs': removed.height,
        'changed_businesses': pl.concat([added.select('sidx'), removed.select('sidx')])['sidx'].n_unique(),
        'non_france_unchanged': True, 'france_score': None})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'fit', 'select', 'evaluate', 'inference'])
    for name in ('base-work', 'work', 'output', 'dataset'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--split', choices=['train', 'test', 'holdout'], default='train')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--threads', type=int, default=30)
    p.add_argument('--rounds', type=int, default=900)
    args = p.parse_args()
    (args.work / 'models').mkdir(parents=True, exist_ok=True)
    {'prepare': prepare, 'fit': fit_models, 'select': select, 'evaluate': evaluate, 'inference': inference}[args.stage](args)


if __name__ == '__main__':
    main()
