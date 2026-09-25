import tempfile
from pathlib import Path
import unittest
import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
from rapidfuzz import process, fuzz
from er_v2.block import (PAIR_SCHEMA, build_target_index, generate, make_keys,
                         make_rescue_keys, merge_candidates, RESCUE_CAPS)
from er_v2.features import STRING_FEATURES, add_block_context, compute, pair_frame, record_frames
from er_v2.metrics import macro_f05
from er_v2.predict import write_lists
from er_v2.train import context_features, fold_expr, predict_frame, stage1_scores, select_matches
from er_v2.runtime import feature_parts


def records():
    return pl.DataFrame({'idx': [0, 1, 2], 'country': ['France', 'US', 'France'],
                         'business_name': ['Martin Bakery', 'Martin Bakery', 'Other'],
                         'name_n': ['martin bakery', 'martin bakery', 'other'],
                         'core_n': ['martin bakery', 'martin bakery', 'other'],
                         'addr_n': ['12 rue hugo', '12 rue hugo', '']}).with_columns(pl.col('idx').cast(pl.UInt32))


class R5Tests(unittest.TestCase):
    def test_country_partition_matches_global_blocking(self):
        df = records()
        for keys, caps in ((make_keys, None), (make_rescue_keys, RESCUE_CAPS)):
            def block(a, b):
                idx = build_target_index(keys(b), len(df), **({'caps': caps} if caps else {}))
                return generate(keys(a), idx, verbose=False).sort('sidx', 'tidx')
            global_pairs = block(df, df)
            partitioned = pl.concat([block(df.filter(pl.col('country') == c), df.filter(pl.col('country') == c))
                                    for c in ('France', 'US')]).sort('sidx', 'tidx')
            assert_frame_equal(global_pairs, partitioned)
            self.assertFalse(global_pairs.filter((pl.col('sidx') == 0) & (pl.col('tidx') == 1)).height)

    def test_rescue_keeps_base_and_deduplicates(self):
        base = pl.DataFrame({'sidx': [0, 0], 'tidx': [1, 2], 'bscore': [3., 2.],
                             'nkeys': [1, 1], 'brank': [1, 2]}, schema=PAIR_SCHEMA)
        rescue = base.with_columns(tidx=pl.Series([2, 3], dtype=pl.UInt32))
        merged = merge_candidates(base, rescue, 64)
        self.assertEqual(merged['tidx'].to_list(), [1, 2, 3])
        self.assertEqual(merged['rescue'].to_list(), [0, 0, 1])
        assert_frame_equal(merged.filter(pl.col('rescue') == 0).drop('rescue'), base)

    def test_string_conversion_optimization_preserves_all_scores(self):
        df = records()
        left, right = record_frames(df, df)
        pairs = pl.DataFrame({'sidx': [0, 2, 2], 'tidx': [1, 0, 2], 'bscore': [3., 2., 1.],
                             'nkeys': [1, 1, 1], 'brank': [1, 1, 2]}, schema=PAIR_SCHEMA)
        attached = pair_frame(add_block_context(pairs), left, right, 2)
        optimized = compute(attached, workers=1)
        for field, name, scorer in STRING_FEATURES:
            expected = process.cpdist(attached[field+'_l'].to_list(), attached[field+'_r'].to_list(),
                                      scorer=scorer, workers=1, dtype=np.float32)
            np.testing.assert_array_equal(optimized[name].to_numpy(), expected)
        for field in ('core', 'addr'):
            expected = process.cpdist(attached[field+'_l'].to_list(), attached[field+'_r'].to_list(),
                                      scorer=fuzz.token_sort_ratio, workers=1, dtype=np.float32)
            np.testing.assert_array_equal(optimized[field+'_tsort'].to_numpy(), expected)

    def test_stage1_scores_each_row_once_and_out_of_fold(self):
        class Model:
            best_iteration = 0
            def __init__(self, value): self.value, self.rows = value, 0
            def inplace_predict(self, x, iteration_range):
                self.rows += len(x)
                self.assert_range = iteration_range
                return np.full(len(x), self.value, np.float32)
        default, alternative = Model(.2), Model(.8)
        with tempfile.TemporaryDirectory() as temp:
            data = pl.DataFrame({'sidx': range(100), 'tidx': range(100), 'f': np.ones(100)})
            data.write_parquet(Path(temp)/'part_000.parquet')
            actual = stage1_scores(Path(temp), {0: alternative, 1: default}, default, ['f'], batch_rows=7)
        expected = np.where(data.select(fold_expr())['fold'].to_numpy() == 0, .8, .2)
        np.testing.assert_allclose(actual['p1'].to_numpy(), expected)
        self.assertEqual(default.rows + alternative.rows, 100)
        self.assertEqual(default.assert_range, (0, 1))

    def test_singletons_duplicates_and_missed_links(self):
        truth = pl.DataFrame({'sidx': [0, 0], 'tidx': [10, 10]})
        pred = pl.DataFrame({'sidx': [1], 'tidx': [11]})
        result = macro_f05(pred, truth, pl.Series([0, 1, 2, 2]))
        self.assertEqual(result['anchors'], 3)
        self.assertAlmostEqual(result['macro_f05'], 1/3)

    def test_empty_decisions_and_outputs_include_unknown_country(self):
        empty = pl.DataFrame(schema={'sidx': pl.UInt32, 'tidx': pl.UInt32, 'p2': pl.Float32})
        for policy in ({'kind': 'threshold', 'threshold': .5},
                       {'kind': 'expected_f05', 'miss': .25, 'floor': .05}):
            self.assertTrue(select_matches(empty, policy).is_empty())
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp)/'matching.tsv'
            s1 = pl.DataFrame({'idx': [0, 1], 'entity_id': ['S1-fr', 'S1-new']})
            write_lists(dest, s1, empty, pl.Series(['S2-a']), 'matched_entity_ids')
            self.assertEqual(dest.read_text().splitlines(),
                             ['source1_entity_id\tmatched_entity_ids', 'S1-fr\t', 'S1-new\t'])

    def test_empty_blocking_and_incomplete_manifest(self):
        keys = make_keys(records().head(0))
        index = build_target_index(make_keys(records()), 3)
        self.assertEqual(dict(generate(keys, index, verbose=False).schema), PAIR_SCHEMA)
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/'_INCOMPLETE').touch()
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                feature_parts(Path(temp))


if __name__ == '__main__':
    unittest.main()
