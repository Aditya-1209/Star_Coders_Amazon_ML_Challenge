"""R15 feature leakage, real classifier/gate, fallback and runner regressions."""
import json
import importlib.metadata
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl

from er_v2 import r10, r15
from er_v2.features import SPLIT_DEPENDENT
from er_v2.folds import fold_expr
from er_v2.r15_features import enrich, alternatives, sibling_features

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_r15


class FeatureTests(unittest.TestCase):
    def fixture(self):
        pairs = pl.DataFrame({'sidx': [0, 0, 0, 1, 2], 'tidx': [0, 1, 3, 0, 2],
            'ce_logit': [5., 3., 4., 8., 2.], 'core_tset': [100., 80., 95., 99., 20.],
            'addr_tset': [100., 10., 90., 0., 0.], 'label': [1, 0, 1, 0, 0]}).with_columns(
                pl.col('sidx', 'tidx').cast(pl.UInt32))
        targets = pl.DataFrame({'idx': [0, 1, 2, 3], 'core_n': ['acme', 'other', 'solo', 'acme'],
                               'addr_n': ['10 main street', '80 west', '', '10 main street']}).with_columns(
                                   pl.col('idx').cast(pl.UInt32))
        return pairs, targets

    def test_signed_winner_margin_excludes_self_and_other_businesses(self):
        pairs, _ = self.fixture()
        actual = alternatives(pairs, 3).sort('sidx', 'tidx')
        self.assertEqual(actual['r15_ce_alternative_margin'].to_list(), [2., -2., None, None, None])
        # Another business gives target 0 a much higher score; it cannot alter
        # these features when graph training exposes fewer source businesses.
        self.assertTrue(actual.head(3).equals(alternatives(pairs.head(3), 3).sort('sidx', 'tidx')))
        tied = pairs.head(3).with_columns(ce_logit=pl.lit(5.))
        self.assertEqual(alternatives(tied, 3)['r15_ce_alternative_margin'].head(2).to_list(), [0., 0.])

    def test_cross_source_support_has_no_self_support_and_missing_is_not_agreement(self):
        pairs, targets = self.fixture()
        actual = sibling_features(pairs, targets, 3, batch_rows=2).sort('sidx', 'tidx')
        self.assertEqual(actual['r15_sibling_ce'].to_list(), [4., 4., 5., None, None])
        self.assertEqual(actual['r15_sibling_number_overlap'].to_list(), [1., 0., 1., None, None])
        self.assertEqual(actual['r15_sibling_name_ratio'][0], 100.)
        self.assertIsNone(actual['r15_sibling_address_ratio'][-1])
        blank = targets.with_columns(addr_n=pl.lit(''))
        self.assertEqual(sibling_features(pairs, blank, 3)['r15_sibling_address_tset'].null_count(), len(pairs))

    def test_evidence_ignores_labels_folds_and_order_and_handles_sparse_sources(self):
        pairs, targets = self.fixture()
        expected = enrich(pairs, targets, 3).sort('sidx', 'tidx')
        actual = enrich(pairs.reverse().with_columns(label=1 - pl.col('label')), targets.reverse(), 3)
        self.assertTrue(expected.drop('label').equals(actual.sort('sidx', 'tidx').drop('label')))
        only = pairs.head(1)
        result = enrich(only, targets, 3)
        self.assertEqual(result['r15_sibling_ce'].null_count(), 1)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            enrich(pl.concat([only, only]), targets, 3)


class PipelineTests(unittest.TestCase):
    def test_parent_preflight_rejects_changed_data_dependencies_and_neural_folds(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = run_r15.parser().parse_args(['--base-work', str(root / 'base'), '--work', str(root / 'new'),
                '--dataset', str(root / 'dataset'), '--validator', str(root / 'validator.py'), '--device', 'cpu', '--reserve-gb', '1'])
            args.work.mkdir()
            for path in run_r15.parent_paths(args):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}')
            for split in ('train', 'test'):
                (args.dataset / split).mkdir(parents=True)
                (args.dataset / split / 'fixture.tsv').write_text('fixture')
                pl.DataFrame({'sidx': [0], 'tidx': [0], 'core_tset': [1.], 'addr_tset': [1.], 'direct': [1]}).write_parquet(
                    args.base_work / f'stage3_{split}.parquet')
            args.validator.write_text('# validator fixture')
            versions = {name: importlib.metadata.version(name) for name in ('polars', 'numpy', 'pyarrow', 'xgboost', 'torch')}
            state = {'status': 'complete', 'identity': {'data': run_r15.fingerprint([args.dataset / 'train', args.dataset / 'test']),
                      'validator': run_r15.fingerprint([args.validator]), 'versions': versions}}
            (args.base_work / 'run.json').write_text(json.dumps(state))
            (args.base_work / 'result.json').write_text(json.dumps({'official_validation': 'PASS', 'metrics': {'local_fold4': {}}}))
            (args.base_work / 'selection.json').write_text(json.dumps({'version': 'r10-ce-ann-1', 'selected': 'r10'}))
            ce_path = args.base_work / 'ce_model/training.json'
            ce = {'train_folds': [0, 1, 8], 'early_stopping_folds': [9]}
            ce_path.write_text(json.dumps(ce))
            (args.base_work / 'neural_e5/finetune.json').write_text(json.dumps({'encoder_folds': [0, 1, 8, 9]}))
            with patch.object(run_r15.shutil, 'disk_usage', return_value=SimpleNamespace(free=50 * 1024**3)):
                self.assertEqual(run_r15.preflight(args), versions)
                ce_path.write_text(json.dumps({**ce, 'train_folds': [0, 1, 4, 8]}))
                with self.assertRaisesRegex(ValueError, 'fold provenance'):
                    run_r15.preflight(args)
                ce_path.write_text(json.dumps(ce))
                state['identity']['versions'] = {**versions, 'polars': 'different'}
                (args.base_work / 'run.json').write_text(json.dumps(state))
                with self.assertRaisesRegex(ValueError, 'Dependencies changed'):
                    run_r15.preflight(args)
                state['identity']['versions'] = versions
                (args.base_work / 'run.json').write_text(json.dumps(state))
                (args.dataset / 'train/fixture.tsv').write_text('changed fixture')
                with self.assertRaisesRegex(ValueError, 'Dataset differs'):
                    run_r15.preflight(args)

    def test_fit_preserves_folds_and_excludes_population_features(self):
        data = pl.DataFrame({'sidx': np.arange(200, dtype=np.uint32)}).with_columns(fold_expr())
        data = data.with_columns(tidx=pl.col('sidx'), label=(pl.col('sidx') % 2).cast(pl.Int8),
                                ce_logit=pl.lit(.2), r15_sibling_ce=pl.lit(.4))
        data = data.with_columns(*[pl.lit(1.).alias(c) for c in SPLIT_DEPENDENT | r15.INCOMPLETE_CONTEXT])
        with TemporaryDirectory() as tmp:
            work, base = Path(tmp) / 'new', Path(tmp) / 'base'
            (work / 'models').mkdir(parents=True)
            base.mkdir()
            (base / 'shift_check.json').write_text('{}')
            calls = []
            class Model:
                best_iteration = 2
                def save_model(self, path):
                    Path(path).write_text('{}')
            def checked_fit(tr, features, va, *rest):
                self.assertEqual(set(tr['fold']), {6, 7})
                self.assertEqual(set(va['fold']), {3})
                self.assertTrue(va.select(r10.half()).to_series().eq(0).all())
                self.assertFalse(set(features) & (SPLIT_DEPENDENT | r15.INCOMPLETE_CONTEXT | {'label', 'fold', 'sidx', 'tidx'}))
                self.assertIn('r15_sibling_ce', features)
                calls.append(1)
                return Model()
            args = SimpleNamespace(work=work, base_work=base, device='cpu', threads=2, rounds=2)
            with patch.object(r15, 'frame_for', return_value=data), patch.object(r15, 'fit', side_effect=checked_fit):
                r15.fit_final(args)
            self.assertEqual(len(calls), 2)

    def test_failed_gate_uses_frozen_r12_decision_not_graph_reference(self):
        pairs = pl.DataFrame({'sidx': [1, 1], 'tidx': [2, 3], 'score': [.8, .6]})
        country = pl.DataFrame({'sidx': [1], 'country': ['France']})
        base_selection = {'selected': 'r10', 'proposal': {'threshold': .75, 'country_thresholds': {}}}
        selection = {'selected': 'r12', 'base_selection': base_selection,
                     'proposal': {'threshold': .1, 'country_thresholds': {}}}
        args = SimpleNamespace(base_work=Path('base'), work=Path('new'))
        self.assertEqual(r15.decide(args, selection, pairs, country)['tidx'].to_list(), [2])
        with TemporaryDirectory() as tmp:
            args.work = Path(tmp)
            (args.work / 'selection.json').write_text(json.dumps(selection))
            with patch.object(r15, 'reference', return_value=(base_selection, pairs)), \
                 patch.object(r15, 'frame_for', side_effect=AssertionError('fallback must not need candidate model')):
                _, scores = r15.chosen(args, 'test')
                self.assertTrue(scores.equals(pairs))

    def test_real_tree_fit_and_prediction_accept_nullable_corroboration(self):
        ids = pl.DataFrame({'sidx': np.arange(300, dtype=np.uint32)}).with_columns(fold_expr())
        data = pl.concat([ids.with_columns(tidx=pl.col('sidx') * 2, label=pl.lit(1),
                            ce_logit=pl.lit(3.), r15_sibling_ce=pl.lit(2.)),
                          ids.with_columns(tidx=pl.col('sidx') * 2 + 1, label=pl.lit(0),
                            ce_logit=pl.lit(-3.), r15_sibling_ce=pl.lit(None, pl.Float64))])
        with TemporaryDirectory() as tmp:
            work, base = Path(tmp) / 'new', Path(tmp) / 'base'
            (work / 'models').mkdir(parents=True)
            base.mkdir()
            (base / 'shift_check.json').write_text('{}')
            args = SimpleNamespace(work=work, base_work=base, device='cpu', threads=2, rounds=8, batch_rows=100)
            with patch.object(r15, 'frame_for', return_value=data):
                r15.fit_final(args)
            actual = r15.predict(args, data, 'mean')
            self.assertEqual(len(actual), len(data))
            self.assertTrue(actual['score'].is_finite().all())
            self.assertGreater(actual['score'][:300].mean(), actual['score'][300:].mean())

    def test_plan_reuses_parent_without_neural_training_and_preserves_gate_order(self):
        plan = run_r15.commands(run_r15.parser().parse_args([]))
        self.assertEqual(list(plan), ['audit', 'features_train', 'features_test', 'fit', 'select', 'evaluate', 'inference', 'validate'])
        self.assertFalse(any('ce_train' in str(cmd) or 'finetune' in str(cmd) for cmd, _ in plan.values()))
        self.assertIn('--check-ids', plan['validate'][0])


if __name__ == '__main__':
    unittest.main()
