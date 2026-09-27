"""Presence leakage, singleton labels, ownership and deadline plan tests."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl

from er_v2 import r10, r16_fast as r16
from er_v2.decision import decide_country
from er_v2.features import SPLIT_DEPENDENT
from er_v2.folds import fold_expr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_r16_fast


class PresenceTests(unittest.TestCase):
    def fixture(self):
        data = pl.DataFrame({'sidx': [0, 0, 1], 'tidx': [1, 2, 3]}).with_columns(
            pl.col('sidx', 'tidx').cast(pl.UInt32),
            *[pl.lit(1., pl.Float32).alias(c) for c in r16.PRESENCE_INPUTS])
        return data.with_columns(ce_logit=pl.Series([4., 2., -1.]),
            r15_sibling_ce=pl.Series([None, 2., None], dtype=pl.Float32)), pl.DataFrame(
                {'sidx': [0, 1, 2]}, schema={'sidx': pl.UInt32})

    def test_business_features_missing_evidence_and_zero_candidates(self):
        frame, anchors = self.fixture()
        result = r16.business_frame(frame, anchors)
        self.assertEqual(result['has_candidates'].to_list(), [1., 1., 0.])
        self.assertEqual(result['ce_top_gap'].to_list(), [2., 0., 0.])
        self.assertEqual(result['r15_sibling_ce_missing'].to_list(), [.5, 1., 0.])
        self.assertFalse(set(result.columns) & (SPLIT_DEPENDENT | {'country', 'tidx', 'label', 'candidate_log_count'}))
        self.assertTrue(result.equals(r16.business_frame(frame.reverse(), anchors.reverse())))
        empty = r16.business_frame(frame.head(0), anchors)
        self.assertEqual(empty['has_candidates'].sum(), 0)
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            r16.business_frame(frame.with_columns(ce_logit=pl.lit(float('nan'))), anchors)

    def test_unretrieved_positive_is_not_labeled_singleton(self):
        frame, anchors = self.fixture()
        business = r16.business_frame(frame, anchors)
        truth = pl.DataFrame({'sidx': [0, 2], 'tidx': [1, 100]}, schema={'sidx': pl.UInt32, 'tidx': pl.UInt32})
        with patch.object(r10, 'truth', return_value=truth):
            result = r16.presence_labels(SimpleNamespace(base_work=Path('base')), business)
        self.assertEqual(result['label'].to_list(), [1, 0, 1])

    def test_presence_filters_before_target_ownership(self):
        scores = pl.DataFrame({'sidx': [0, 1], 'tidx': [3, 3], 'score': [.99, .9]})
        presence = pl.DataFrame({'sidx': [0, 1], 'presence': [.01, .99]})
        country = pl.DataFrame({'sidx': [0, 1], 'country': ['US', 'US']})
        kept = decide_country(r16.eligible(scores, presence, .5), .7, country, {}, 'score')
        self.assertEqual(kept['sidx'].to_list(), [1])
        with self.assertRaisesRegex(ValueError, 'Missing'):
            r16.eligible(scores, presence.head(1), .5)

    def test_failed_gate_uses_exact_parent_and_no_new_features(self):
        selection = {'selected': 'r12', 'base_selection': {'selected': 'r10',
            'proposal': {'threshold': .75, 'country_thresholds': {}}}}
        scores = pl.DataFrame({'sidx': [0, 0], 'tidx': [1, 2], 'score': [.8, .7]})
        country = pl.DataFrame({'sidx': [0], 'country': ['France']})
        with TemporaryDirectory() as tmp:
            args = SimpleNamespace(base_work=Path('base'), work=Path(tmp))
            (args.work / 'selection.json').write_text(json.dumps(selection))
            with patch.object(r16, 'reference', return_value=(selection['base_selection'], scores)), \
                 patch.object(r16, 'frame_for', side_effect=AssertionError('fallback must not use new features')):
                _, actual = r16.chosen(args, 'test')
            self.assertTrue(actual.equals(scores))
            self.assertEqual(r16.decide(args, selection, scores, country)['tidx'].to_list(), [1])

    def test_presence_optimizer_excludes_gate_holdout_and_ids(self):
        ids = pl.DataFrame({'sidx': np.arange(300, dtype=np.uint32)}).with_columns(fold_expr())
        frame = ids.with_columns(tidx=pl.col('sidx'), label=(pl.col('sidx') % 2).cast(pl.Int8),
            *[pl.lit(1., pl.Float32).alias(c) for c in r16.PRESENCE_INPUTS])
        truth = frame.filter(pl.col('sidx') % 2 == 0).select('sidx', 'tidx')
        with TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / 'models').mkdir()
            (work / 'models.json').write_text('{}')
            args = SimpleNamespace(base_work=work, work=work, rounds=3, device='cpu', threads=2)
            class Model:
                best_iteration = 1
                def save_model(self, path):
                    Path(path).write_text('{}')
            def checked(tr, features, va, *rest):
                self.assertEqual(set(tr['fold']), {6, 7})
                self.assertEqual(set(va['fold']), {3})
                self.assertTrue(va.select(r10.half()).to_series().eq(0).all())
                self.assertFalse(set(features) & {'sidx', 'tidx', 'label', 'fold'})
                return Model()
            with patch.object(r16.r15, 'fit_final'), patch.object(r16, 'frame_for', return_value=frame), \
                 patch.object(r10, 'anchors', return_value=ids.select('sidx')), \
                 patch.object(r10, 'truth', return_value=truth), patch.object(r16, 'fit', side_effect=checked):
                r16.fit_final(args)
            self.assertEqual(json.loads((work / 'models.json').read_text())['presence_protocol']['train_folds'], [6, 7])

    def test_plan_skips_neural_work_and_checks_presence_model(self):
        plan = run_r16_fast.commands(run_r16_fast.parser().parse_args([]))
        self.assertEqual(list(plan), ['audit', 'features_train', 'features_test', 'fit', 'select', 'evaluate', 'inference', 'validate'])
        self.assertIn('er_v2.r16_fast', plan['fit'][0])
        self.assertIn(Path('work/r16_fast/models/presence.json'), plan['fit'][1])
        self.assertFalse(any('finetune' in str(c) or 'ce_train' in str(c) for c, _ in plan.values()))
        self.assertIn('--check-ids', plan['validate'][0])


if __name__ == '__main__':
    unittest.main()
