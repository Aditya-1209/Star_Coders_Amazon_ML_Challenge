"""R16 model isolation, singleton decisions, fallback and offline integration."""
import json
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import polars as pl
from er_v2 import r16, r10
from er_v2.folds import fold_expr
from er_v2.r13_evidence import FEATURES, INCOMPLETE_TARGET_CONTEXT

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_r16
import run_r10


def rows(ids):
    return pl.DataFrame({'sidx': np.asarray(ids, np.uint32)}).with_columns(
        tidx=pl.col('sidx'), **{c: pl.lit(.5, pl.Float32) for c in r16.PRESENCE_INPUTS}).with_columns(fold_expr())


class R16Tests(unittest.TestCase):
    def test_business_features_are_label_free_subset_stable_and_include_no_candidates(self):
        frame = rows([0, 0, 2]).with_columns(tidx=pl.Series([1, 2, 3], dtype=pl.UInt32),
                    ce_logit=pl.Series([3., 1., -2.]), p2=pl.Series([None, .8, .3]),
                    name_num_conflict=pl.Series([True, False, False]))
        anchors = pl.DataFrame({'sidx': np.arange(4, dtype=np.uint32)})
        actual = r16.business_frame(frame, anchors)
        self.assertEqual(actual['sidx'].to_list(), [0, 1, 2, 3])
        self.assertEqual(actual.schema['sidx'], pl.UInt32)
        self.assertEqual(actual.filter(pl.col('sidx') == 1)['candidate_log_count'].item(), 0)
        self.assertEqual(actual['ce_top_gap'].to_list(), [2., 0., 0., 0.])
        self.assertTrue(actual.equals(r16.business_frame(frame.reverse().with_columns(label=pl.lit(1)), anchors.reverse())))
        subset = r16.business_frame(frame.filter(pl.col('sidx') == 0), anchors.head(1))
        self.assertTrue(subset.equals(actual.head(1)))
        empty = r16.business_frame(frame.head(0), anchors)
        self.assertEqual(empty['sidx'].to_list(), [0, 1, 2, 3])
        self.assertTrue(empty['candidate_log_count'].eq(0).all())
        with self.assertRaisesRegex(ValueError, 'Invalid presence'):
            r16.business_frame(frame.with_columns(ce_logit=pl.lit(float('nan'))), anchors)
        self.assertFalse(set(actual.columns) & {'label', 'tidx', 'country', 'score'})

    def test_feature_families_keep_a_r12_reference_and_isolate_new_context(self):
        frame = rows([0, 1]).with_columns(**{c: pl.lit(.5) for c in FEATURES},
            **{c: pl.lit(.5) for c in INCOMPLETE_TARGET_CONTEXT}, label=pl.lit(0))
        families = r16.model_features(frame, 'enhanced')
        self.assertFalse(set(families['reference']) & set(FEATURES))
        self.assertTrue(INCOMPLETE_TARGET_CONTEXT <= set(families['reference']))
        self.assertFalse(set(families['enhanced']) & INCOMPLETE_TARGET_CONTEXT)
        self.assertTrue(set(FEATURES) <= set(families['enhanced']))
        self.assertFalse(set(families['enhanced']) & {'label', 'fold', 'sidx', 'tidx'})

    def test_presence_filter_precedes_exclusive_ownership_and_calibration_keeps_singletons(self):
        scores = pl.DataFrame({'sidx': [0, 1, 2], 'tidx': [10, 10, 11], 'score': [.99, .95, .97]}).with_columns(
            pl.col('sidx', 'tidx').cast(pl.UInt32))
        country = pl.DataFrame({'sidx': np.arange(4, dtype=np.uint32), 'country': ['France'] * 4})
        presence = country.select('sidx').with_columns(presence=pl.Series([.02, .99, .01, .01]))
        truth = scores.filter(pl.col('sidx') == 1).select('sidx', 'tidx')
        proposal = r16.calibrated(scores, presence, .1, truth, country, 'reference', 0.)
        chosen = r16.decide(scores, presence, proposal, country)
        self.assertEqual(chosen.select('sidx', 'tidx').rows(), [(1, 10)])
        self.assertEqual(proposal['fold3A'], 1.)  # includes three correct singletons
        with self.assertRaisesRegex(ValueError, 'Missing/non-finite'):
            r16.eligible(scores, presence.head(1), .1)

    def test_presence_training_uses_full_truth_and_never_fits_holdout_or_gate_labels(self):
        country = pl.DataFrame({'sidx': np.arange(400, dtype=np.uint32), 'country': ['US'] * 400}).with_columns(fold_expr())
        expected_ids = country.filter(pl.col('fold').is_in([6, 7]))['sidx']
        # Positive business 1 has no surviving pairs; retain its full-truth label.
        positive = country.filter(pl.col('fold').is_in([6, 7])).head(1)['sidx'].item()
        frame = rows(country['sidx']).filter(pl.col('sidx') != positive).with_columns(
            label=(pl.col('sidx') % 2).cast(pl.Int8), **{c: pl.lit(.5) for c in FEATURES})
        truth = country.filter((pl.col('sidx') % 2 == 0) | (pl.col('sidx') == positive)).select('sidx').with_columns(tidx=pl.col('sidx'))
        calls = []
        class Model:
            best_iteration = 1
            def save_model(self, path): Path(path).write_text('{}')
        def fitting(tr, features, va, *args):
            self.assertTrue(set(tr['fold']) <= {6, 7})
            self.assertEqual(set(va['fold']), {3})
            self.assertTrue(va.select(r10.half())[:, 0].eq(0).all())
            self.assertNotIn('label', features)
            calls.append(tr)
            return Model()
        with TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / 'models').mkdir()
            args = SimpleNamespace(work=work, rounds=2, threads=1, device='cpu')
            with patch.object(r16, 'frame_for', return_value=frame), patch.object(r10, 'anchors', return_value=country), \
                 patch.object(r10, 'truth', return_value=truth), patch.object(r16, 'fit', side_effect=fitting):
                r16.fit_final(args)
            self.assertEqual(len(calls), 5)
            self.assertEqual(set(calls[-1]['sidx']), set(expected_ids))
            self.assertEqual(calls[-1].filter(pl.col('sidx') == positive)['label'].item(), 1)
            self.assertEqual(calls[-1].filter(pl.col('sidx') == positive)['candidate_log_count'].item(), 0)

    def test_selection_falls_back_on_ties_and_never_reads_fold4_labels(self):
        country = pl.DataFrame({'sidx': np.arange(500, dtype=np.uint32), 'country': ['US'] * 500}).filter(fold_expr() == 3)
        frame = rows(country['sidx'])
        scores = frame.select('sidx', 'tidx').with_columns(score=pl.lit(.9))
        presence = country.select('sidx').with_columns(presence=pl.lit(.99))
        def truth(args, folds):
            self.assertEqual(folds, [3])
            return scores.select('sidx', 'tidx')
        with TemporaryDirectory() as tmp:
            args = SimpleNamespace(work=Path(tmp), minimum_gain=.0002)
            with patch.object(r10, 'anchors', return_value=country), patch.object(r10, 'truth', side_effect=truth), \
                 patch.object(r16, 'frame_for', return_value=frame), patch.object(r16, 'predict_pair', return_value=scores), \
                 patch.object(r16, 'predict_presence', return_value=presence):
                with redirect_stdout(io.StringIO()):
                    r16.select(args)
            result = json.loads((args.work / 'selection.json').read_text())
            self.assertEqual(result['selected'], 'reference')
            self.assertFalse(result['gate']['passed'])
            self.assertEqual(result['gate']['gain'], 0)
            self.assertEqual(result['proposal']['presence_cutoff'], 0)

    def test_singleton_improvement_must_survive_the_separate_gate(self):
        country = pl.DataFrame({'sidx': np.arange(4000, dtype=np.uint32), 'country': ['US'] * 4000}).filter(fold_expr() == 3)
        frame = rows(country['sidx'])
        scores = frame.select('sidx', 'tidx').with_columns(score=pl.lit(.9))
        truth = scores.filter(pl.col('sidx') % 3 != 0).select('sidx', 'tidx')
        for passes in (True, False):
            good = (pl.col('sidx') % 3 != 0)
            # The second scenario improves A but reverses the business signal on B.
            present = good if passes else pl.when(r10.half() == 0).then(good).otherwise(~good)
            presence = country.select('sidx').with_columns(presence=pl.when(present).then(.99).otherwise(.01))
            with self.subTest(passes=passes), TemporaryDirectory() as tmp:
                args = SimpleNamespace(work=Path(tmp), minimum_gain=.0002)
                with patch.object(r10, 'anchors', return_value=country), patch.object(r10, 'truth', return_value=truth), \
                     patch.object(r16, 'frame_for', return_value=frame), patch.object(r16, 'predict_pair', return_value=scores), \
                     patch.object(r16, 'predict_presence', return_value=presence), redirect_stdout(io.StringIO()):
                    r16.select(args)
                result = json.loads((args.work / 'selection.json').read_text())
                self.assertEqual(result['proposal']['fold3A'], 1.)
                self.assertEqual(result['selected'], 'r16' if passes else 'reference')
                self.assertEqual(result['gate']['passed'], passes)

    def test_cloud_plan_has_one_ce_no_mining_new_models_and_16_hour_limit(self):
        args = run_r16.parser([]).parse_args([])
        plan = run_r16.commands(args)
        self.assertEqual((args.work, args.max_hours, args.ce_token_cache_gb), (Path('work/r16'), 16, 12))
        self.assertEqual(args.target_leaderboard, .99)
        self.assertNotIn('--mine-stage2', plan['ce_train'][0])
        self.assertIn('--precision', plan['ce_train'][0])
        self.assertEqual(sum(name == 'ce_train' for name in plan), 1)
        self.assertEqual(len(plan['fit'][1]), 6)
        self.assertLess(list(plan).index('select'), list(plan).index('evaluate'))
        self.assertIn('er_v2.r16', plan['inference'][0])
        self.assertEqual(run_r10.parser().parse_args([]).max_hours, 24)


import test_r10


class R16PipelineTest(test_r10.R10PipelineTest):
    r13 = True
    r16 = True


if __name__ == '__main__':
    unittest.main()
