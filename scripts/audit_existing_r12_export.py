"""Read-only audit of the already-created R12 confidence export and raw dataset."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code/business_entity_resolution/src'))
from er_v2.decision import decide_country
from er_v2.folds import fold_expr
from er_v2.metrics import macro_f05, by_country
from er_v2.predict import write_lists


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def raw_records(dataset, split, side):
    return pl.read_csv(dataset / split / f'{split}_source{side}.tsv', separator='\t',
        quote_char=None, infer_schema=False, columns=['entity_id', 'country']).with_row_index('idx')


def load_maps(dataset, split):
    s1 = raw_records(dataset, split, 1).rename({'idx': 'sidx'})
    s2 = raw_records(dataset, split, 2)
    s3 = raw_records(dataset, split, 3).with_columns(pl.col('idx') + len(s2))
    targets = pl.concat([s2, s3]).rename({'idx': 'tidx'})
    assert s1['entity_id'].n_unique() == len(s1), 'Duplicate source IDs in organizer data'
    assert targets['entity_id'].n_unique() == len(targets), 'Ambiguous target source IDs'
    return s1, targets, len(s2)


def audit_table(path, anchors, targets, n_source2, fold=None):
    pf = pq.ParquetFile(path)
    rows = pf.metadata.num_rows
    keys = np.empty(rows, np.uint64)
    covered = np.zeros(len(anchors), bool)
    cursor = 0
    score_min, score_max, score_sum = 1., 0., 0.
    source2_pairs = 0
    for batch in pf.iter_batches(batch_size=250_000):
        frame = pl.from_arrow(batch)
        assert frame.schema == {'sidx': pl.UInt32, 'tidx': pl.UInt32, 'score': pl.Float32,
                                'sid': pl.String, 'tid': pl.String}
        assert sum(frame.null_count().row(0)) == 0, 'Null export value'
        assert frame['score'].is_finite().all() and frame['score'].is_between(0, 1).all(), 'Invalid score'
        assert frame['sidx'].max() < len(anchors) and frame['tidx'].max() < len(targets), 'Out-of-range index'
        assert (anchors['entity_id'].gather(frame['sidx']) == frame['sid']).all(), 'Source ID/index mismatch'
        assert (targets['entity_id'].gather(frame['tidx']) == frame['tid']).all(), 'Target ID/index mismatch'
        assert (anchors['country'].gather(frame['sidx']) == targets['country'].gather(frame['tidx'])).all(), 'Cross-country pair'
        if fold is not None:
            assert frame.select(fold_expr())['fold'].eq(fold).all(), 'Wrong labeled partition'
        sid, tid = frame['sidx'].to_numpy(), frame['tidx'].to_numpy()
        covered[sid] = True
        keys[cursor:cursor + len(frame)] = (sid.astype(np.uint64) << 32) | tid.astype(np.uint64)
        cursor += len(frame)
        source2_pairs += int((tid < n_source2).sum())
        score_min = min(score_min, float(frame['score'].min()))
        score_max = max(score_max, float(frame['score'].max()))
        score_sum += frame['score'].cast(pl.Float64).sum()
    assert cursor == rows
    keys.sort()
    duplicates = int(np.count_nonzero(keys[1:] == keys[:-1]))
    assert duplicates == 0, 'Duplicate candidate keys'
    expected = np.ones(len(anchors), bool) if fold is None else anchors.select(fold_expr())['fold'].to_numpy() == fold
    assert not np.any(covered & ~expected)
    result = {'rows': rows, 'row_groups': pf.metadata.num_row_groups,
        'all_scores_finite_and_in_range': True, 'all_ids_match_organizer_row_indices': True,
        'all_pairs_within_country': True, 'duplicate_pairs': duplicates,
        'expected_businesses': int(expected.sum()), 'businesses_with_candidates': int(covered.sum()),
        'businesses_without_candidates': int(np.count_nonzero(expected & ~covered)),
        'source2_pairs': source2_pairs, 'source3_pairs': rows - source2_pairs,
        'score_min': score_min, 'score_max': score_max, 'score_mean': score_sum / rows}
    del keys, covered
    gc.collect()
    print(path.name, json.dumps(result), flush=True)
    return result


def truth_for(dataset, anchors, targets):
    raw = pl.read_csv(dataset / 'train/train_ground_truth.tsv', separator='\t',
                      infer_schema=False, quote_char=None)
    raw = (raw.join(anchors.select('sidx', 'entity_id'), left_on='source1_entity_id', right_on='entity_id')
        .select('sidx', tid=pl.col('matched_entity_ids').str.split(','))
        .explode('tid', empty_as_null=True).drop_nulls().filter(pl.col('tid') != ''))
    truth = targets.select('tidx', 'entity_id').join(raw, left_on='entity_id', right_on='tid').select('sidx', 'tidx')
    assert len(truth) == len(raw), 'Ground-truth ID not represented in organizer target mapping'
    assert truth.n_unique() == len(truth), 'Duplicate ground truth'
    return truth


def compare(actual, expected):
    assert set(actual) == set(expected)
    for key in actual:
        assert abs(actual[key] - expected[key]) < 1e-12, (key, actual[key], expected[key])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-dir', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads((args.export_dir / 'selection.json').read_text())
    published = json.loads((args.export_dir / 'metrics.json').read_text())
    assert selection == published['selection']
    assert selection['selected'] == 'r10' and selection['version'] == 'r10-ce-ann-1'
    proposal = selection['proposal']
    report = {'archive_score_tables': {}, 'reproduced_metrics': {},
              'versions': {'polars': pl.__version__, 'numpy': np.__version__},
              'status': 'incomplete', 'fit_or_threshold_tuning_performed': False}
    for split in ('train', 'test'):
        print('Loading organizer ID mappings:', split, flush=True)
        anchors, targets, n_s2 = load_maps(args.dataset, split)
        report[split + '_source_counts'] = {'source1': len(anchors), 'source2': n_s2, 'source3': len(targets) - n_s2}
        if split == 'train':
            for fold in (3, 4):
                path = args.export_dir / f'r12_fold{fold}.parquet'
                report['archive_score_tables'][path.name] = audit_table(path, anchors, targets, n_s2, fold)
                scope = anchors.filter(fold_expr() == fold)
                truth = truth_for(args.dataset, scope, targets)
                scores = pl.read_parquet(path, columns=['sidx', 'tidx', 'score'])
                for name, current in ([('3A', scope.filter(pl.col('sidx').hash(seed=1010) % 2 == 0)),
                                       ('3B', scope.filter(pl.col('sidx').hash(seed=1010) % 2 == 1))]
                                      if fold == 3 else [('fold4', scope)]):
                    local = scores.join(current.select('sidx'), on='sidx', how='semi')
                    decisions = decide_country(local, proposal['threshold'], current.select('sidx', 'country'),
                                               proposal['country_thresholds'], 'score')
                    actual = macro_f05(decisions, truth, current['sidx'])
                    countries = by_country(decisions, truth, current.select('sidx', 'country'))
                    if name == '3A':
                        assert abs(actual['macro_f05'] - proposal['fold3A']) < 1e-12
                    elif name == '3B':
                        assert set(countries) == set(selection['gate']['candidate_by_country'])
                        for country, values in countries.items(): compare(values, selection['gate']['candidate_by_country'][country])
                    else:
                        compare(actual, published['local_fold4'])
                        assert set(countries) == set(published['by_country'])
                        for country, values in countries.items(): compare(values, published['by_country'][country])
                        oracle = macro_f05(local.select('sidx', 'tidx').join(truth, on=['sidx', 'tidx']), truth, current['sidx'])
                        compare(oracle, published['candidate_oracle_fold4'])
                        report['fold4_candidate_oracle'] = oracle
                    report['reproduced_metrics'][name] = {'overall': actual, 'by_country': countries,
                        'matches_published_result': True}
                    print(name, json.dumps(report['reproduced_metrics'][name]), flush=True)
                del truth, scores, local, decisions, scope, current
                gc.collect()
        else:
            path = args.export_dir / 'r12_test.parquet'
            report['archive_score_tables'][path.name] = audit_table(path, anchors, targets, n_s2)
            assert not proposal['country_thresholds'], 'Audit prefilter requires global frozen cutoff'
            scores = pl.scan_parquet(path).select('sidx', 'tidx', 'score').filter(
                pl.col('score') >= proposal['threshold']).collect(engine='streaming')
            decisions = decide_country(scores, proposal['threshold'], anchors.select('sidx', 'country'), {}, 'score')
            assert decisions['tidx'].n_unique() == len(decisions)
            # The verified reference is committed on r16, so a single-branch
            # clone does not need Git objects from the separate r12 branch.
            expected = json.loads((ROOT / 'reports/r12_confidence_export_verified.json').read_text())['test_reproduction']
            with TemporaryDirectory(dir=args.report.parent) as scratch:
                path = Path(scratch) / 'matching_results.tsv'
                write_lists(path, anchors.select(idx=pl.col('sidx'), entity_id=pl.col('entity_id')),
                            decisions, targets['entity_id'], 'matched_entity_ids')
                actual = sha256(path)
                assert actual == expected['sha256'], ('Test TSV checksum mismatch', actual)
                assert path.stat().st_size == expected['bytes']
                report['test_reproduction'] = {'sha256': actual, 'bytes': path.stat().st_size,
                    'byte_identical_to_published_submission': True, 'matching_pairs': len(decisions),
                    'source1_rows': len(anchors), 'empty_rows': len(anchors) - decisions['sidx'].n_unique()}
            print('Test reproduction:', json.dumps(report['test_reproduction']), flush=True)
            del scores, decisions
        del anchors, targets
        gc.collect()
    report.update(status='complete', sufficient_r12_scores_for_r14_comparison=True,
        additional_r12_export_needed=False,
        dependencies=['Same organizer data for full truth and all-business/source membership',
                      'R14 predictions with entity IDs and compatible training provenance'],
        limitation='Exact producing script and original model checkpoints are not in the archive; '
                   'this audit verifies all exported rows and reproduces the frozen reported decisions/results.')
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print('Complete:', args.report, flush=True)


if __name__ == '__main__':
    main()
