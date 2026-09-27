"""Specialist scope, sibling ambiguity, fold isolation and unchanged fallback."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import polars as pl
from er_v2 import r10, r15_rescue as rescue
from er_v2.features import SPLIT_DEPENDENT
from er_v2.folds import fold_expr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_r15_rescue


class RescueTests(unittest.TestCase):
    def test_confidence_uses_other_source_ambiguity_not_self_or_labels(self):
        frame = pl.DataFrame({'sidx': [0, 0, 0, 1, 1, 1], 'tidx': [0, 3, 4, 0, 3, 4],
            'ce_logit': [4., 4., 1., 4., 4., 4.], 'label': [0, 1, 0, 1, 0, 0]}).with_columns(
                pl.col('sidx', 'tidx').cast(pl.UInt32),
                *[pl.lit(95.).alias('r15_sibling_' + n) for n in
                    ('ce', 'name_ratio', 'name_tset', 'address_ratio', 'address_tset', 'number_overlap')])
        actual = rescue.trusted_support(frame, 3).sort('sidx', 'tidx')
        self.assertEqual(actual['rescue_donor_margin'].to_list(), [3., None, None, 0., None, None])
        self.assertEqual(actual['rescue_trusted_name_tset'].to_list(), [95., 95., 95., None, 95., 95.])
        self.assertTrue(actual.drop('label').equals(rescue.trusted_support(
            frame.reverse().with_columns(label=1-pl.col('label')), 3).sort('sidx', 'tidx').drop('label')))
        absent = frame.head(1)
        self.assertIsNone(rescue.trusted_support(absent, 3)['rescue_trusted_name_tset'][0])

    def test_only_specialist_candidate_scores_change(self):
        base = pl.DataFrame({'sidx': [0, 0, 1], 'tidx': [1, 2, 3], 'score': [.9, .4, .8]})
        current = pl.DataFrame({'sidx': [0], 'tidx': [2], 'score': [.8]})
        actual = rescue.conditional_blend(base, current, .5)
        np.testing.assert_allclose(actual['score'], [.9, .6, .8])
        self.assertTrue(actual.select('sidx', 'tidx').equals(base.select('sidx', 'tidx')))
        with self.assertRaisesRegex(ValueError, 'missing from'):
            rescue.conditional_blend(base, current.with_columns(tidx=pl.lit(99)), .5)

    def test_fit_only_missing_address_6_7_and_3a(self):
        ids = pl.DataFrame({'sidx': np.arange(300, dtype=np.uint32)}).with_columns(fold_expr())
        frame = pl.concat([ids.with_columns(tidx=pl.col('sidx') * 2, addr_len_r=pl.lit(0), label=pl.lit(1), ce_logit=pl.lit(3.)),
                           ids.with_columns(tidx=pl.col('sidx') * 2 + 1, addr_len_r=pl.lit(20), label=pl.lit(0), ce_logit=pl.lit(-3.))])
        frame = frame.with_columns(r15_sibling_name_tset=pl.lit(100.), rescue_trusted_name_tset=pl.lit(100.),
            *[pl.lit(1.).alias(c) for c in SPLIT_DEPENDENT | rescue.r15.INCOMPLETE_CONTEXT])
        with TemporaryDirectory() as tmp:
            work, base = Path(tmp)/'new', Path(tmp)/'base'
            (work/'models').mkdir(parents=True)
            base.mkdir()
            (base/'shift_check.json').write_text('{}')
            args = SimpleNamespace(base_work=base, work=work, rounds=3, device='cpu', threads=2)
            class Model:
                best_iteration = 1
                def save_model(self, path):
                    Path(path).write_text('{}')
            def checked(tr, features, va, *more):
                self.assertEqual(set(tr['fold']), {6,7})
                self.assertTrue(tr['addr_len_r'].eq(0).all())
                self.assertTrue(va['addr_len_r'].eq(0).all())
                self.assertEqual(set(va['fold']), {3})
                self.assertTrue(va.select(r10.half()).to_series().eq(0).all())
                self.assertFalse(set(features) & (SPLIT_DEPENDENT | rescue.r15.INCOMPLETE_CONTEXT | {'sidx','tidx','label','fold','r15_sibling_name_tset'}))
                self.assertIn('rescue_trusted_name_tset', features)
                return Model()
            with patch.object(rescue, 'frame_for', return_value=frame), patch.object(rescue, 'fit', side_effect=checked):
                rescue.fit_final(args)

    def test_real_specialist_prediction_is_limited_to_missing_candidates(self):
        ids = pl.DataFrame({'sidx': np.arange(300, dtype=np.uint32)}).with_columns(fold_expr())
        frame = pl.concat([ids.with_columns(tidx=pl.col('sidx')*3, addr_len_r=pl.lit(0), label=pl.lit(1), ce_logit=pl.lit(3.)),
            ids.with_columns(tidx=pl.col('sidx')*3+1, addr_len_r=pl.lit(0), label=pl.lit(0), ce_logit=pl.lit(-3.)),
            ids.with_columns(tidx=pl.col('sidx')*3+2, addr_len_r=pl.lit(20), label=pl.lit(0), ce_logit=pl.lit(-1.))])
        with TemporaryDirectory() as tmp:
            work, base = Path(tmp)/'new', Path(tmp)/'base'
            (work/'models').mkdir(parents=True)
            base.mkdir()
            (base/'shift_check.json').write_text('{}')
            args = SimpleNamespace(base_work=base, work=work, rounds=8, device='cpu', threads=2, batch_rows=100)
            with patch.object(rescue, 'frame_for', return_value=frame):
                rescue.fit_final(args)
            actual = rescue.predict(args, frame, 'mean')
            self.assertEqual(len(actual), 600)
            self.assertGreater(actual['score'][:300].mean(), actual['score'][300:].mean())

    def test_fallback_does_not_touch_specialist(self):
        with TemporaryDirectory() as tmp:
            work=Path(tmp)
            selection={'selected':'r12','base_selection':{'selected':'r10','proposal':{'threshold':.75,'country_thresholds':{}}}}
            (work/'selection.json').write_text(json.dumps(selection))
            scores=pl.DataFrame({'sidx':[0,0],'tidx':[1,2],'score':[.8,.6]})
            args=SimpleNamespace(work=work,base_work=Path('base'))
            with patch.object(rescue,'reference',return_value=(selection['base_selection'],scores)), \
                 patch.object(rescue,'frame_for',side_effect=AssertionError('fallback must skip new model')):
                _, actual=rescue.chosen(args,'test')
            self.assertTrue(actual.equals(scores))

    def test_plan_reuses_neural_work(self):
        plan=run_r15_rescue.commands(run_r15_rescue.parser().parse_args([]))
        self.assertIn('er_v2.r15_rescue',plan['fit'][0])
        self.assertEqual(list(plan)[-2:],['inference','validate'])
        self.assertFalse(any('ce_train' in str(c) or 'finetune' in str(c) for c,_ in plan.values()))


if __name__ == '__main__':
    unittest.main()
