"""Small regression tests for r4. No challenge data or GPU is required."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl

from er_v2.block import PAIR_SCHEMA, generate
from er_v2.decision import best_per_target
from er_v2.graph import HOP_SCHEMA, SUPPORT_SCHEMA, expand, support_features
from er_v2.metrics import macro_f05
from er_v2.runtime import feature_parts
from er_v2.stage3 import meta_thr
from er_v2.train import predict_frame, validation_sample


def frame(values, schema):
    return pl.DataFrame(values, schema=schema)


class R4RegressionTests(unittest.TestCase):
    def test_exclusivity_ties_do_not_depend_on_input_order(self):
        scores = pl.DataFrame({"sidx": [2, 1, 3], "tidx": [9, 9, 9], "p2": [0.9, 0.9, float("nan")]})
        for data in (scores, scores.reverse()):
            self.assertEqual(best_per_target(data, "p2")["sidx"].to_list(), [1])

    def test_no_shared_block_keys_has_a_typed_empty_result(self):
        source = frame({"idx": [0], "key": [1]}, {"idx": pl.UInt32, "key": pl.UInt64})
        target = frame({"idx": [1], "key": [2], "w": [1.0]},
                       {"idx": pl.UInt32, "key": pl.UInt64, "w": pl.Float32})
        actual = generate(source, target, verbose=False)
        self.assertTrue(actual.is_empty())
        self.assertEqual(dict(actual.schema), PAIR_SCHEMA)

    def test_no_graph_anchors_is_supported(self):
        anchors = pl.DataFrame(schema={"sidx": pl.UInt32, "a": pl.UInt32, "pa": pl.Float32})
        self.assertEqual(dict(expand(anchors, pl.DataFrame(), pl.DataFrame()).schema), HOP_SCHEMA)
        pairs = frame({"sidx": [0], "tidx": [1]}, {"sidx": pl.UInt32, "tidx": pl.UInt32})
        self.assertEqual(dict(support_features(pairs, anchors, pl.DataFrame()).schema), SUPPORT_SCHEMA)

    def test_hop_limit_is_ten_even_when_self_is_not_retrieved(self):
        anchors = frame({"sidx": [0], "a": [100], "pa": [0.9]},
                        {"sidx": pl.UInt32, "a": pl.UInt32, "pa": pl.Float32})
        keys = frame({"idx": [100], "key": [7]}, {"idx": pl.UInt32, "key": pl.UInt64})
        index = frame({"idx": list(range(20)), "key": [7] * 20, "w": [1.0] * 20},
                      {"idx": pl.UInt32, "key": pl.UInt64, "w": pl.Float32})
        result = expand(anchors, keys, index)
        self.assertEqual(len(result), 10)
        self.assertEqual(sorted(result["tidx"].to_list()), list(range(10)))

    def test_missing_address_is_not_perfect_support(self):
        pairs = frame({"sidx": [0], "tidx": [1]}, {"sidx": pl.UInt32, "tidx": pl.UInt32})
        anchors = frame({"sidx": [0], "a": [2], "pa": [0.9]},
                        {"sidx": pl.UInt32, "a": pl.UInt32, "pa": pl.Float32})
        text = frame({"idx": [1, 2], "core_r": ["alpha", "alpha"], "addr_r": ["", ""], "cc_r": ["alpha", "alpha"]},
                     {"idx": pl.UInt32, "core_r": pl.String, "addr_r": pl.String, "cc_r": pl.String})
        result = support_features(pairs, anchors, text, workers=1)
        self.assertEqual(result["sup_addr_ratio"][0], 0.0)
        self.assertEqual(result["sup_addr_valid"][0], 0)
        self.assertEqual(dict(result.schema), SUPPORT_SCHEMA)

    def test_metric_treats_pairs_and_anchors_as_sets(self):
        pairs = pl.DataFrame({"sidx": [0, 0], "tidx": [1, 1]})
        result = macro_f05(pairs, pairs, pl.Series([0, 0, 1]))
        self.assertEqual(result["anchors"], 2)
        self.assertEqual(result["macro_f05"], 1.0)
        self.assertEqual(result["pair_precision"], 1.0)

    def test_small_validation_sets_do_not_request_two_million_rows(self):
        sample = pl.DataFrame({"label": [0, 1, 0]})
        self.assertEqual(len(validation_sample(sample, seed=42)), 3)
        with self.assertRaisesRegex(ValueError, "no candidate pairs"):
            validation_sample(sample.head(0), seed=42)

    def test_batched_scoring_and_empty_frames(self):
        class Booster:
            best_iteration = 0

            def __init__(self):
                self.calls = []

            def inplace_predict(self, x, iteration_range):
                self.calls.append((len(x), iteration_range))
                return x[:, 0] / 10

        model = Booster()
        data = pl.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0, 5.0]})
        np.testing.assert_allclose(predict_frame(model, data, ["feature"], 2), [0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertEqual(model.calls, [(2, (0, 1)), (2, (0, 1)), (1, (0, 1))])
        self.assertEqual(len(predict_frame(model, data.head(0), ["feature"], 2)), 0)
        self.assertEqual(len(model.calls), 3)

    def test_threshold_comes_from_selected_model_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "metrics.json").write_text(json.dumps({"threshold": 0.42}), encoding="utf-8")
            self.assertEqual(meta_thr(path), 0.42)

    def test_incomplete_and_stale_feature_shards_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            part = path / "part_000.parquet"
            part.write_bytes(b"fixture")
            (path / "_INCOMPLETE").touch()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                feature_parts(path)
            (path / "_INCOMPLETE").unlink()
            record = {"name": part.name, "bytes": part.stat().st_size, "mtime_ns": part.stat().st_mtime_ns}
            (path / "manifest.json").write_text(json.dumps({"parts": [record]}), encoding="utf-8")
            self.assertEqual(feature_parts(path), [part])
            (path / "part_001.parquet").write_bytes(b"stale")
            with self.assertRaisesRegex(ValueError, "stale shards"):
                feature_parts(path)


if __name__ == "__main__":
    unittest.main()
