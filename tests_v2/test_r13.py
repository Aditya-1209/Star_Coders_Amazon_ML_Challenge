"""R13 leakage, hard-negative, sibling support, profile and integration checks."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import polars as pl

from er_v2.folds import fold_expr
from er_v2.r10 import frame_for
from er_v2.r10_ce import samples, TRAIN_FOLDS
from er_v2.r13_evidence import trusted_anchors, SiblingIndex, FEATURES, INCOMPLETE_TARGET_CONTEXT
from er_v2.r13_diagnostics import error_report

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_r10
import run_r13


def frame(data):
    return pl.DataFrame(data).with_columns(pl.col('sidx', 'tidx').cast(pl.UInt32))


class R13Tests(unittest.TestCase):
    def test_mining_excludes_all_positives_and_other_folds_with_bounded_diverse_negatives(self):
        source = pl.DataFrame({'idx': np.arange(100, dtype=np.uint32)})
        candidates = frame({'sidx': np.repeat(np.arange(100), 14),
                            'tidx': np.repeat(np.arange(100) * 20, 14) + np.tile(np.arange(14), 100)})
        # Seven positives: neither a four-positive cap nor candidate-label column may hide one.
        target = candidates.filter(pl.col('tidx') % 20 < 7)
        candidates = candidates.with_columns(brank=(pl.col('tidx') % 20).cast(pl.UInt16))
        ann = candidates.with_columns(nrank=(20 - pl.col('tidx') % 20).cast(pl.UInt16))
        scores = candidates.with_columns(p2=(pl.col('tidx') % 20 / 14).cast(pl.Float32), label=pl.lit(1))
        with TemporaryDirectory() as tmp:
            work = Path(tmp)
            candidates.write_parquet(work / 'cands_train.parquet')
            ann.write_parquet(work / 'ncands_train.parquet')
            scores.write_parquet(work / 'stage2_train.parquet')
            with patch('er_v2.r10_ce.load_split', return_value=(source, pl.DataFrame())), \
                 patch('er_v2.r10_ce.ground_truth_pairs', return_value=target):
                actual = samples(work, work, TRAIN_FOLDS, 100, mine_stage2=True)
                # Held-out labels/scores and row order cannot alter CE training pairs.
                scores.with_columns(p2=pl.when(fold_expr().is_in([3, 4])).then(.999).otherwise(pl.col('p2')),
                                    label=pl.lit(0)).reverse().write_parquet(work / 'stage2_train.parquet')
                self.assertTrue(actual.equals(samples(work, work, TRAIN_FOLDS, 100, mine_stage2=True)))
                with self.assertRaisesRegex(ValueError, 'out-of-sample'):
                    samples(work, work, [2, 4], 100, mine_stage2=True)
        self.assertTrue(set(actual.select(fold_expr())['fold']) <= set(TRAIN_FOLDS))
        self.assertTrue(actual.filter(pl.col('label') == 1).select('sidx', 'tidx').equals(
            target.filter(fold_expr().is_in(TRAIN_FOLDS)).sort('sidx', 'tidx')))
        neg = actual.filter(pl.col('label') == 0)
        self.assertEqual(neg.group_by('sidx').len()['len'].unique().to_list(), [5])
        for ids in neg.group_by('sidx').agg((pl.col('tidx') % 20).alias('targets'))['targets']:
            self.assertTrue({7, 11, 12, 13} <= set(ids))  # lexical, ANN, two hardest model errors
        np.testing.assert_allclose(actual.group_by('sidx').agg(pl.col('w').sum())['w'], 12.)

    def test_anchor_ownership_ties_and_strong_competitors_are_rejected(self):
        scores = frame({'sidx': [0, 1, 0, 2, 0, 0, 0], 'tidx': [0, 0, 1, 1, 2, 3, 4],
                        'p2': [.99, .99, .99, .93, .999, .995, .985]})
        actual = trusted_anchors(scores, k=2)
        self.assertEqual(actual['tidx'].to_list(), [2, 3])
        self.assertTrue(actual.equals(trusted_anchors(scores.reverse(), k=2)))
        # A competitor outside the graph folds must still suppress an unsafe anchor.
        self.assertEqual(trusted_anchors(scores.filter(pl.col('sidx') == 0))['tidx'].to_list(), [2, 3, 0])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            trusted_anchors(pl.concat([scores, scores.head(1)]))

    def test_sibling_excludes_self_handles_absence_and_preserves_negative_cosine(self):
        emb = np.array([[1., 0.], [0., 1.], [-1., 0.], [.8, .6]], dtype=np.float32)
        anchors = frame({'sidx': [0, 0, 1], 'tidx': [0, 1, 2], 'p2': [.999, .99, .995]})
        index = SiblingIndex(anchors, emb, 3)
        pairs = frame({'sidx': [0, 0, 1, 1, 2], 'tidx': [0, 3, 2, 0, 3], 'ncos': [.9, .5, .8, .1, .5]})
        actual = index.features(pairs, batch_rows=1)
        self.assertTrue(actual.equals(index.features(pairs, batch_rows=100)))
        np.testing.assert_allclose(actual['sib_count'], [1, 2, 0, 1, 0])
        np.testing.assert_allclose(actual['sib_cos_max'], [0, .8, 0, -1, 0], atol=1e-6)
        np.testing.assert_allclose(actual['sib_cos_gain'], [-.9, .3, 0, -1.1, 0], atol=1e-6)
        self.assertEqual(actual.row(2)[2:], (0.,) * len(FEATURES))
        self.assertEqual(actual.row(4)[2:], (0.,) * len(FEATURES))
        empty = SiblingIndex(anchors.head(0), emb, 3).features(pairs)
        self.assertTrue(all(empty[c].eq(0).all() for c in FEATURES))

    def test_evidence_join_is_keyed_and_missing_values_fail_closed(self):
        rows = frame({'sidx': [0, 1, 2], 'tidx': [1, 2, 3], 'direct': [1, 1, 1],
            'sup_both_max': [0., 0., 0.], 'sup_addr_valid': [0, 0, 0], 'sup_name_tset': [0., 0., 0.]})
        evidence = rows.select('sidx', 'tidx').with_columns([pl.lit(float(i)).alias(c) for i, c in enumerate(FEATURES)])
        with TemporaryDirectory() as tmp:
            work = Path(tmp)
            rows.write_parquet(work / 'stage3_train.parquet')
            rows.select('sidx', 'tidx').with_columns(ce_logit=pl.lit(.4)).write_parquet(work / 'ce_train.parquet')
            evidence.reverse().write_parquet(work / 'r13_evidence_train.parquet')
            args = SimpleNamespace(work=work, r13_features=True)
            self.assertTrue(frame_for(args, 'train').select(evidence.columns).equals(evidence))
            evidence.head(2).write_parquet(work / 'r13_evidence_train.parquet')
            with self.assertRaisesRegex(ValueError, 'Missing/non-finite R13'):
                frame_for(args, 'train')

    def test_final_fitting_removes_incomplete_target_context_but_keeps_siblings(self):
        from er_v2 import r10
        rows = pl.DataFrame({'sidx': np.arange(200, dtype=np.uint32)}).with_columns(fold_expr())
        rows = rows.with_columns(tidx=pl.col('sidx'), label=(pl.col('sidx') % 2).cast(pl.Int8),
            ce_logit=pl.lit(.1), ce_rank_s=pl.lit(1), sib_cos_max=pl.lit(.8),
            **{c: pl.lit(.2) for c in INCOMPLETE_TARGET_CONTEXT})
        class Model:
            best_iteration = 1
            def save_model(self, path):
                Path(path).write_text('{}')
        def fit(train, features, valid, *more):
            self.assertTrue(set(train['fold']) <= {6, 7})
            self.assertTrue(valid.select(r10.half())[:, 0].eq(0).all())
            self.assertFalse(set(features) & INCOMPLETE_TARGET_CONTEXT)
            self.assertTrue({'ce_logit', 'ce_rank_s', 'sib_cos_max'} <= set(features))
            return Model()
        with TemporaryDirectory() as tmp:
            w = Path(tmp)
            (w / 'models').mkdir()
            with patch.object(r10, 'frame_for', return_value=rows), patch.object(r10, 'fit', side_effect=fit):
                r10.fit_final(SimpleNamespace(work=w, rounds=2, threads=1, device='cpu', r13_features=True))
            self.assertEqual(json.loads((w / 'r10_models.json').read_text())['version'], 'r13-mined-siblings-1')

    def test_error_report_distinguishes_retrieval_rejection_and_source3_address_offsets(self):
        target = frame({'sidx': [0, 0, 1], 'tidx': [0, 2, 3]})
        scores = frame({'sidx': [0, 0, 1], 'tidx': [0, 2, 1]})
        chosen = frame({'sidx': [0, 1], 'tidx': [0, 1]})
        with TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / 'norm').mkdir()
            for side, addresses in ((2, ['full', 'wrong']), (3, ['', 'full'])):
                pl.DataFrame({'addr_n': addresses}).write_parquet(work / f'norm/train_source{side}.parquet')
            report = error_report(work, scores, chosen, target)
        self.assertEqual((report['missed_pairs'], report['retrieved_but_rejected'], report['not_retrieved']), (2, 1, 1))
        self.assertEqual(report['rejected_with_missing_address'], 1)
        self.assertEqual(report['by_target_address']['present']['false_positive_pairs'], 1)
        self.assertIsNone(report['by_target_address']['missing']['pair_precision'])

    def test_profiles_propagate_r13_training_features_deadlines_and_website_target(self):
        for profile in ('vm', 'desktop'):
            argv = ['--profile', profile]
            args = run_r13.parser(argv).parse_args(argv)
            plan = run_r10.commands(args)
            self.assertEqual(args.target_leaderboard, .985)
            self.assertEqual(args.work, Path('work/r13'))
            self.assertEqual(args.ce_batch, 64 if profile == 'vm' else 16)
            self.assertEqual(args.ce_checkpointing, profile == 'desktop')
            self.assertIn('--mine-stage2', plan['ce_train'][0])
            self.assertIn('--record-competition', plan['train'][0])
            for split in ('train', 'test'):
                self.assertLess(list(plan).index('graph_' + split), list(plan).index('evidence_' + split))
                self.assertEqual(len(plan['evidence_' + split][1]), 2)
            for stage in ('fit', 'select', 'evaluate', 'inference'):
                self.assertIn('--r13-features', plan[stage][0])
            self.assertLess(list(plan).index('select'), list(plan).index('evaluate'))
        old = run_r10.commands(run_r10.parser().parse_args([]))
        self.assertNotIn('evidence_train', old)
        self.assertNotIn('--mine-stage2', old['ce_train'][0])
        argv = ['--profile', 'desktop', '--ce-batch', '8', '--max-hours', '7']
        override = run_r13.parser(argv).parse_args(argv)
        self.assertEqual((override.ce_batch, override.max_hours), (8, 7))


import test_r10


class R13PipelineTest(test_r10.R10PipelineTest):
    r13 = True


if __name__ == '__main__':
    unittest.main()
