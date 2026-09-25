"""Accuracy-path checks for the teammate's PC; no dataset/model/GPU needed.

These tests were written but deliberately not executed on the author's laptop.
"""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl

from er_v2.block import PAIR_SCHEMA, generate
from er_v2.decision import NO_MATCH_THRESHOLD, blend_scores, tune_blend, tune_threshold
from er_v2.features import add_block_context, compute, pair_frame, record_frames
from er_v2.metrics import macro_f05
from er_v2.train import STAGE1_GROUPS, decide, fold_expr, stage1_scores


class AccuracyTests(unittest.TestCase):
    def test_channel_union_recovers_a_name_match_without_losing_combined_hit(self):
        source = pl.DataFrame({"idx": [0, 0], "key": [10, 20]},
                              schema={"idx": pl.UInt32, "key": pl.UInt64})
        target = pl.DataFrame({"idx": [1, 2], "key": [10, 20], "w": [4.0, 9.0], "kind": ["n", "w"]},
                              schema={"idx": pl.UInt32, "key": pl.UInt64, "w": pl.Float32, "kind": pl.String})
        original = generate(source, target, top_k=1, verbose=False)
        diverse = generate(source, target, top_k=1, name_k=1, address_k=1, verbose=False)
        self.assertEqual(original["tidx"].to_list(), [2])
        self.assertEqual(set(diverse["tidx"]), {1, 2})
        self.assertTrue(set(original["tidx"]).issubset(set(diverse["tidx"])))
        self.assertEqual(dict(diverse.schema), PAIR_SCHEMA)
        self.assertEqual(diverse.select("sidx", "tidx").n_unique(), len(diverse))

    def test_exact_cutoff_matches_brute_force_including_ties_and_singletons(self):
        rng = np.random.default_rng(41)
        anchors = pl.Series("sidx", range(10), dtype=pl.UInt32)
        for _ in range(8):
            pred = pl.DataFrame({"sidx": np.repeat(np.arange(8, dtype=np.uint32), 5),
                                 "tidx": rng.integers(0, 20, 40, dtype=np.uint32),
                                 "p2": rng.choice([0.0, 0.2, 0.4, 0.7, 1.0], 40)}).unique(["sidx", "tidx"])
            truth = pl.DataFrame({"sidx": [0, 0, 1, 3, 4, 8], "tidx": [1, 2, 3, 4, 5, 19]})
            actual, threshold = tune_threshold(pred, truth, anchors)
            # Zero-probability edges are never selected by the optimized rule.
            cuts = [NO_MATCH_THRESHOLD] + sorted(set(pred.filter(pl.col("p2") > 0)["p2"]), reverse=True)
            candidates = [(macro_f05(decide(pred, cutoff), truth, anchors)["macro_f05"], cutoff)
                          for cutoff in cuts]
            expected = max(value for value, _ in candidates)
            self.assertAlmostEqual(actual, expected, places=12)
            self.assertAlmostEqual(macro_f05(decide(pred, threshold), truth, anchors)["macro_f05"], expected)

    def test_cutoff_can_find_a_gain_between_old_grid_points(self):
        pred = pl.DataFrame({"sidx": [0, 1], "tidx": [0, 1], "p2": [0.613, 0.611]})
        truth = pl.DataFrame({"sidx": [0], "tidx": [0]})
        metric, threshold = tune_threshold(pred, truth, pl.Series([0, 1]))
        self.assertEqual(threshold, 0.613)
        self.assertAlmostEqual(metric, 1.0)

    def test_no_match_rule_rejects_even_a_false_probability_one(self):
        pred = pl.DataFrame({"sidx": [0], "tidx": [0], "p2": [1.0]})
        truth = pl.DataFrame(schema={"sidx": pl.UInt32, "tidx": pl.UInt32})
        metric, threshold = tune_threshold(pred, truth, pl.Series([0, 1]))
        self.assertEqual(metric, 1.0)
        self.assertTrue(decide(pred, threshold).is_empty())
        empty_metric, _ = tune_threshold(pred.head(0), truth, pl.Series([0, 1]))
        self.assertEqual(empty_metric, 1.0)

    def test_graph_selection_keeps_stage2_when_graph_scores_are_worse(self):
        pred = pl.DataFrame({"sidx": [0, 1, 1], "tidx": [0, 1, 2],
                             "p2": [0.9, 0.1, None], "p3": [0.1, 0.9, 0.95]})
        truth = pl.DataFrame({"sidx": [0], "tidx": [0]})
        selection = tune_blend(pred, truth, pl.Series([0, 1]))
        self.assertEqual(selection["stage3_weight"], 0.0)
        selected = decide(blend_scores(pred, selection["stage3_weight"]), selection["threshold"], "score")
        self.assertEqual(selected["tidx"].to_list(), [0])

    def test_graph_selection_can_recover_a_hop_only_true_match(self):
        pred = pl.DataFrame({"sidx": [0, 0, 1], "tidx": [0, 1, 2],
                             "p2": [0.95, None, 0.2], "p3": [0.95, 0.9, 0.1]})
        truth = pl.DataFrame({"sidx": [0, 0], "tidx": [0, 1]})
        selection = tune_blend(pred, truth, pl.Series([0, 1]))
        selected = decide(blend_scores(pred, selection["stage3_weight"]), selection["threshold"], "score")
        self.assertGreater(selection["stage3_weight"], 0)
        self.assertEqual(set(selected["tidx"]), {0, 1})

    def test_stage1_training_rows_use_only_the_complementary_model(self):
        class ConstantBooster:
            best_iteration = 0

            def __init__(self, probability):
                self.probability = probability

            def inplace_predict(self, x, iteration_range):
                return np.full(len(x), self.probability, dtype=np.float32)

        rows = pl.DataFrame({"sidx": range(100), "tidx": range(100), "f": [1.0] * 100}).with_columns(
            pl.col("sidx", "tidx").cast(pl.UInt32))
        a, b = ConstantBooster(0.2), ConstantBooster(0.8)
        held_out = {fold: b for fold in STAGE1_GROUPS[0]}
        held_out.update({fold: a for fold in STAGE1_GROUPS[1]})
        with tempfile.TemporaryDirectory() as folder:
            rows.write_parquet(Path(folder) / "part_000.parquet")
            actual = stage1_scores(Path(folder), held_out, [a, b], ["f"], batch_rows=7).with_columns(fold_expr())
        for group, expected in ((STAGE1_GROUPS[0], 0.8), (STAGE1_GROUPS[1], 0.2), ((2, 3, 4, 5, 6, 7), 0.5)):
            values = actual.filter(pl.col("fold").is_in(group))["p1"].to_numpy()
            self.assertGreater(len(values), 0)
            np.testing.assert_allclose(values, expected)

    def test_numeric_conflicts_postcode_zeroes_and_missing_evidence(self):
        source = pl.DataFrame({"idx": [0], "country": ["US"], "business_name": ["Acme 5"],
                               "business_address": ["10 Main Street 02110"], "name_n": ["acme s"],
                               "core_n": ["acme s"], "addr_n": ["10 main st 2110"]}).with_columns(pl.col("idx").cast(pl.UInt32))
        target = pl.DataFrame({"idx": [0, 1], "country": ["US", "US"], "business_name": ["Acme 3", "Acme 3"],
                               "business_address": ["20 Main Street 02110", ""], "name_n": ["acme e", "acme e"],
                               "core_n": ["acme e", "acme e"], "addr_n": ["20 main st 2110", ""]}).with_columns(pl.col("idx").cast(pl.UInt32))
        left, right = record_frames(source, target)
        candidates = pl.DataFrame({"sidx": [0, 0], "tidx": [0, 1], "bscore": [2.0, 1.0],
                                   "nkeys": [1, 1], "brank": [1, 2]}, schema=PAIR_SCHEMA)
        result = compute(pair_frame(add_block_context(candidates), left, right, 2), workers=1).sort("tidx")
        self.assertTrue(result["name_num_conflict"][0])
        self.assertTrue(result["house_conflict"][0])
        self.assertTrue(result["postcode_equal"][0])
        self.assertFalse(result["postcode_conflict"][1])
        self.assertFalse(result["addr_both_present"][1])
        self.assertEqual(result["addr_ratio"][1], 0.0)
        self.assertAlmostEqual(result["name_frequency_r"][0], np.log1p(2), places=5)


if __name__ == "__main__":
    unittest.main()
