"""Frozen fusion export, identity alignment, and fail-closed provenance checks."""
from contextlib import redirect_stdout
import importlib.metadata
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import polars as pl
import xgboost as xgb
from er_v2 import r10
from er_v2.folds import fold_expr
from er_v2.metrics import macro_f05, by_country

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import export_r12_evidence as exporter


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root, selected='r10'):
    work, data, output = root / 'parent', root / 'data', root / 'original_output'
    (work / 'norm').mkdir(parents=True)
    (work / 'models').mkdir()
    (data / 'train').mkdir(parents=True)
    n = 300
    for split in ('train', 'test'):
        for side in (1, 2, 3):
            pl.DataFrame({'idx': np.arange(n, dtype=np.uint32),
                'entity_id': [f'{split}-S{side}-{i}' for i in range(n)],
                'country': ['US' if i % 2 else 'India' for i in range(n)]}).write_parquet(
                    work / f'norm/{split}_source{side}.parquet')
        pairs = pl.DataFrame({'sidx': np.tile(np.arange(n - 1, dtype=np.uint32), 2),
                             'tidx': np.arange(2 * (n - 1), dtype=np.uint32)})
        pairs = pairs.with_columns(fold_expr(), direct=pl.lit(1), sup_both_max=pl.lit(0.),
                                  sup_addr_valid=pl.lit(0), sup_name_tset=pl.lit(0.))
        pairs.write_parquet(work / f'stage3_{split}.parquet')
        pairs.select('sidx', 'tidx', ce_logit=pl.when(pl.col('tidx') < n - 1).then(4.).otherwise(-4.)).write_parquet(
            work / f'ce_{split}.parquet')
        pairs.select('sidx', 'tidx', 'fold', score=pl.lit(.05, pl.Float32)).write_parquet(
            work / ('eval_preds_stage3.parquet' if split == 'train' else 'test_preds_stage3.parquet'))
    # Include a truth match for the business without any candidates.
    pl.DataFrame({'source1_entity_id': [f'train-S1-{i}' for i in range(n)],
                  'matched_entity_ids': [f'train-S2-{i}' for i in range(n)]}).write_csv(
                      data / 'train/train_ground_truth.tsv', separator='\t')
    for kind in ('pair', 'business'):
        matrix = xgb.DMatrix(np.array([[-4.], [4.]] * 100, dtype=np.float32),
                              label=[0, 1] * 100, feature_names=['ce_logit'])
        model = xgb.train({'objective': 'binary:logistic', 'max_depth': 1, 'nthread': 1}, matrix, 8)
        model.save_model(work / f'models/r10_{kind}.json')
    save(work / 'r10_models.json', {'features': ['ce_logit']})
    save(work / 'models/stage3_metrics.json', {'threshold': .5, 'country_thresholds': {}})
    save(work / 'ce_model/training.json', {'train_folds': [0, 1, 8], 'early_stopping_folds': [9]})
    save(work / 'neural_e5/finetune.json', {'encoder_folds': [0, 1, 8, 9]})
    selection = {'version': 'r10-ce-ann-1', 'selected': selected,
                 'proposal': {'model': 'mean', 'weight': 1., 'threshold': .5, 'country_thresholds': {}}}
    save(work / 'selection.json', selection)
    args = SimpleNamespace(work=work, dataset=data, output=output, threads=1, device='cpu', batch_rows=31)
    country = r10.anchors(args).filter(fold_expr() == 3)
    _, scores = r10.chosen_scores(args, 'train', 3)
    truth = r10.truth(args, [3])
    for half in (0, 1):
        scope = country.filter(r10.half() == half)
        pred = r10.final_decision(args, selection, scores.join(scope.select('sidx'), on='sidx', how='semi'), scope)
        if half == 0:
            selection['proposal']['fold3A'] = macro_f05(pred, truth, scope['sidx'])['macro_f05']
        else:
            selection['gate'] = {'candidate_by_country': by_country(pred, truth, scope),
                                 'baseline_by_country': by_country(pred, truth, scope)}
    save(work / 'selection.json', selection)
    country = r10.anchors(args).filter(fold_expr() == 4)
    _, scores = r10.chosen_scores(args, 'train', 4)
    truth = r10.truth(args, [4])
    pred = r10.final_decision(args, selection, scores, country)
    metrics = {'selection': selection, 'local_fold4': macro_f05(pred, truth, country['sidx']),
               'by_country': by_country(pred, truth, country)}
    save(work / 'metrics.json', metrics)
    r10.inference(args)
    save(work / 'result.json', {'status': 'complete', 'official_validation': 'PASS', 'metrics': metrics,
        'output_sha256': {name: exporter.digest(output / name) for name in ('candidate_pairs.tsv', 'matching_results.tsv')}})
    records = lambda paths: [{'path': str(p), 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in paths]
    save(work / 'run.json', {'status': 'complete', 'identity': {'code': 'fixture',
        'versions': {name: importlib.metadata.version(name) for name in ('polars', 'numpy', 'xgboost', 'pyarrow')},
        'data': records([data / 'train/train_ground_truth.tsv'])},
        'completed': {'fixture': {'outputs': records([p for p in work.rglob('*') if p.is_file()])}}})
    return SimpleNamespace(base_work=work, dataset=data, output=root / 'export', threads=1, device='cpu')


class R12ExportTests(unittest.TestCase):
    def test_actual_fusion_and_reference_exports_reproduce_frozen_decisions(self):
        for selected in ('r10', 'reference'):
            with self.subTest(selected=selected), TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
                args = fixture(Path(tmp), selected)
                before = exporter.snapshot([p for p in args.base_work.rglob('*') if p.is_file()])
                exporter.export(args)
                manifest = exporter.read_json(args.output / 'manifest.json')
                self.assertTrue(manifest['checks']['test']['official_tsv_hashes_match'])
                rows = pl.read_parquet(args.output / 'test_scores.parquet')
                self.assertEqual(len(rows), 598)
                self.assertEqual(rows['selected'].sum(), 299 if selected == 'r10' else 0)
                self.assertEqual(pl.read_parquet(args.output / 'test_source1_ids.parquet').height, 300)
                self.assertEqual(set(rows['target_source']), {2, 3})
                if selected == 'r10':
                    self.assertGreater(rows['score'].max(), .5)  # graph-only export would be .05 everywhere
                self.assertEqual(before, exporter.snapshot(map(Path, before)))
                self.assertFalse((args.output / '_INCOMPLETE').exists())
                for name, info in manifest['files'].items():
                    self.assertEqual(exporter.digest(args.output / name), info['sha256'])
                with self.assertRaises(FileExistsError):
                    exporter.export(args)

    def test_entity_alignment_is_by_keys_and_never_silently_drops_pairs(self):
        with TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            anchors, targets = exporter.mappings(args.base_work, 'test')
            rows = pl.DataFrame({'sidx': [0, 1], 'tidx': [300, 0], 'score': [.8, .9]}).with_columns(
                pl.col('sidx', 'tidx').cast(pl.UInt32))
            actual = exporter.canonical(rows, anchors.reverse(), targets.reverse()).sort('sidx')
            self.assertEqual(actual['target_entity_id'].to_list(), ['test-S3-0', 'test-S2-0'])
            self.assertEqual(actual['target_source'].to_list(), [3, 2])
            for bad in (pl.concat([rows, rows.head(1)]), rows.with_columns(tidx=pl.lit(999, pl.UInt32)),
                        rows.with_columns(score=pl.lit(float('nan')))):
                with self.assertRaises(ValueError):
                    exporter.canonical(bad, anchors, targets)

    def test_changed_parent_fails_before_creating_export(self):
        with TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            (args.base_work / 'r10_models.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'artifact no longer matches'):
                exporter.export(args)
            self.assertFalse(args.output.exists())

    def test_bad_cached_test_scores_cannot_claim_a_validated_export(self):
        with TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            args = fixture(Path(tmp))
            cache = args.base_work / 'r10_test_predictions.parquet'
            pl.read_parquet(cache).with_columns(score=pl.lit(.05, pl.Float32)).write_parquet(cache)
            # Even if a producer incorrectly recorded graph-only scores as its
            # final cache, the official matching TSV hash must still reject it.
            state = exporter.read_json(args.base_work / 'run.json')
            for item in state['completed']['fixture']['outputs']:
                if item['path'] == str(cache):
                    item.update(bytes=cache.stat().st_size, mtime_ns=cache.stat().st_mtime_ns)
            save(args.base_work / 'run.json', state)
            with self.assertRaisesRegex(ValueError, 'officially validated matching_results'):
                exporter.export(args)
            self.assertFalse((args.output / 'manifest.json').exists())
            self.assertTrue((args.output / '_INCOMPLETE').exists())

    def test_holdout_metric_mismatch_is_rejected(self):
        values = dict(anchors=10, macro_f05=.99, pair_precision=1., pair_recall=.96)
        with self.assertRaisesRegex(ValueError, 'metric mismatch'):
            exporter.check_metrics(values, {**values, 'pair_recall': .95}, 'fold4')


if __name__ == '__main__':
    unittest.main()
