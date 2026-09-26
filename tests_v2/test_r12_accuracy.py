"""Regressions for training coverage, missing fields and CE feature consistency."""
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl
import torch

from er_v2.folds import fold_expr
from er_v2.r10 import frame_for
from er_v2.r10_ce import PairTokens, samples, score_logits, TRAIN_FOLDS
from er_v2.stage3 import competition_features


class AccuracyTests(unittest.TestCase):
    def test_graph_record_competition_retains_businesses_outside_graph_folds(self):
        graph = pl.DataFrame({'sidx': [1, 2], 'tidx': [10, 10],
                              'core_tset': [90., 80.], 'addr_tset': [0., 20.]})
        pool = pl.concat([graph, pl.DataFrame({'sidx': [3], 'tidx': [10],
                                               'core_tset': [100.], 'addr_tset': [50.]})])
        actual = competition_features(graph, record_competition=True, record_pool=pool)
        self.assertEqual(actual['rc_n_claim'].to_list(), [3, 3])
        self.assertEqual(actual['rc_name_gap_other'].to_list(), [-10., -20.])
        self.assertEqual(actual['rc_addr_gap_other'].to_list(), [-50., -30.])
        self.assertEqual(len(actual), len(graph))
        # Fold selection cannot erase a competing name; external labels are ignored.
        partial = competition_features(graph.head(1), record_competition=True,
                                       record_pool=pool.with_columns(label=pl.lit(1)))
        self.assertTrue(partial.equals(actual.head(1)))

    def test_ce_context_is_identical_when_loading_different_folds(self):
        ids = pl.DataFrame({'sidx': np.arange(100, dtype=np.uint32)}).with_columns(fold_expr())
        rows = pl.concat([ids.filter(pl.col('fold') == f).head(1) for f in (3, 4, 6, 7)])
        rows = rows.with_columns(tidx=pl.lit(10, pl.UInt32), label=pl.lit(0, pl.Int8),
            direct=pl.lit(1), sup_both_max=pl.lit(0.), sup_addr_valid=pl.lit(0), sup_name_tset=pl.lit(0.))
        scores = rows.select('sidx', 'tidx').with_columns(ce_logit=pl.Series([4., 1., 3., 2.]))
        with TemporaryDirectory() as tmp:
            w = Path(tmp)
            rows.write_parquet(w / 'stage3_train.parquet')
            scores.write_parquet(w / 'ce_train.parquet')
            args = SimpleNamespace(work=w)
            full = frame_for(args, 'train')
            for folds in ([3], [4], [6, 7, 3]):
                subset = frame_for(args, 'train', folds)
                self.assertTrue(subset.equals(full.filter(pl.col('fold').is_in(folds))))
            held = frame_for(args, 'train', [4])
            self.assertEqual(held['ce_rank_t'].item(), 4)
            self.assertEqual(held['ce_gap_t'].item(), 3.)
            # Changing labels cannot change score-only competition.
            rows.with_columns(label=1 - pl.col('label')).write_parquet(w / 'stage3_train.parquet')
            self.assertTrue(frame_for(args, 'train').drop('label').equals(full.drop('label')))

    def test_ce_retains_every_positive_and_never_uses_other_fold_labels(self):
        source = pl.DataFrame({'idx': np.arange(100, dtype=np.uint32)})
        truth = pl.DataFrame({'sidx': np.repeat(np.arange(100, dtype=np.uint32), 6),
                              'tidx': np.repeat(np.arange(100, dtype=np.uint32) * 10, 6)
                                      + np.tile(np.arange(6, dtype=np.uint32), 100)})
        candidates = pl.concat([truth, source.select(sidx='idx', tidx=pl.col('idx') * 10 + 7)]).with_columns(
            brank=(pl.col('tidx') % 10 + 1).cast(pl.UInt16))
        with TemporaryDirectory() as tmp:
            work = Path(tmp)
            candidates.write_parquet(work / 'cands_train.parquet')
            candidates.rename({'brank': 'nrank'}).write_parquet(work / 'ncands_train.parquet')
            with patch('er_v2.r10_ce.load_split', return_value=(source, pl.DataFrame())), \
                 patch('er_v2.r10_ce.ground_truth_pairs', return_value=truth):
                actual = samples(work, work, TRAIN_FOLDS, 100)
        self.assertTrue(set(actual.select(fold_expr())['fold']) <= set(TRAIN_FOLDS))
        expected = truth.filter(fold_expr().is_in(TRAIN_FOLDS)).sort('sidx', 'tidx')
        positives = actual.filter(pl.col('label') == 1).select('sidx', 'tidx')
        self.assertTrue(positives.equals(expected))
        self.assertTrue((actual.filter(pl.col('label') == 0)['tidx'] % 10).eq(7).all())
        np.testing.assert_allclose(actual.group_by('sidx').agg(pl.col('w').sum())['w'], 7.)

    def test_target_dropout_matches_naturally_blank_template_before_and_after_swap(self):
        class Batch(list):
            def to(self, device):
                return self
        class Tokenizer:
            cls_token_id, sep_token_id = 101, 102
            model_input_names = ['input_ids', 'attention_mask', 'token_type_ids']
            def num_special_tokens_to_add(self, pair):
                return 3
            def pad(self, examples, **kwargs):
                return Batch(examples)
        cache = object.__new__(PairTokens)
        cache.tokenizer = Tokenizer()
        cache.arrays = {'s1': np.array([[11, 102, 31]]),
                        'tg': np.array([[21, 102, 41], [21, 102, 0]])}
        cache.lengths = {'s1': np.array([3]), 'tg': np.array([3, 2])}
        pair = pl.DataFrame({'sidx': [0], 'tidx': [0]})
        natural = pair.with_columns(tidx=pl.lit(1))
        for swap in (False, True):
            augmented = cache.batch(pair, 'cpu', swap=swap, drop_address=True)
            self.assertEqual(augmented, cache.batch(natural, 'cpu', swap=swap))
            self.assertIn(31, augmented[0]['input_ids'])
            self.assertNotIn(41, augmented[0]['input_ids'])
        batch = cache.batch(pl.concat([pair, pair]), 'cpu', drop_address=[True, False])
        self.assertNotIn(41, batch[0]['input_ids'])
        self.assertIn(41, batch[1]['input_ids'])

    def test_length_bucketing_and_oom_retry_preserve_all_pair_scores(self):
        frame = pl.DataFrame({'sidx': np.arange(6, dtype=np.uint32), 'tidx': np.arange(6, dtype=np.uint32)})
        class Cache:
            def pair_lengths(self, rows):
                return np.array([11, 2, 7, 1, 4, 3])[rows['sidx'].to_numpy()]
            def batch(self, rows, device):
                return {'ids': torch.tensor(rows['sidx'].to_list())}
        class Model:
            def __init__(self, limit=100):
                self.limit, self.seen, self.attempts = limit, [], []
            def __call__(self, ids):
                self.attempts.append(len(ids))
                if len(ids) > self.limit:
                    raise torch.cuda.OutOfMemoryError('synthetic capacity boundary')
                self.seen.extend(ids.tolist())
                return SimpleNamespace(logits=(ids * 10).float())
        model, cache = Model(), Cache()
        result = score_logits(model, cache, frame, 'cpu', 3)
        np.testing.assert_array_equal(result, np.arange(6) * 10)
        self.assertEqual(model.seen, [3, 1, 5, 4, 2, 0])
        model, cache = Model(3), Cache()
        # Simulate a CUDA allocation failure without requiring a GPU in CI.
        with patch('torch.autocast', return_value=nullcontext()), patch('torch.cuda.empty_cache'):
            result = score_logits(model, cache, frame, 'cuda', 8)
            np.testing.assert_array_equal(result, np.arange(6) * 10)
            self.assertEqual(model.attempts, [6, 3, 3])
            self.assertEqual(sorted(model.seen), list(range(6)))
            self.assertEqual(cache.score_batch_limit, 3)
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                score_logits(Model(0), Cache(), frame.head(1), 'cuda', 8)


if __name__ == '__main__':
    unittest.main()
