from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import polars as pl

from er_v2.block import build_target_index, make_keys, make_phonetic_keys
from er_v2.indexing import build_index
from er_v2.matrix import FrameIter
from er_v2.name_features import compute_name_features
from er_v2.run_block import load_split
from er_v2.train import ContextIndex, fit, predict_frame
from er_v2.decision import decide_country, tune_country_thresholds


class R7Tests(unittest.TestCase):
    def test_country_cutoffs_keep_global_fallback_for_france(self):
        anchors = pl.DataFrame({"sidx": [0, 1, 2, 3], "country": ["US", "India", "France", "France"]})
        pred = pl.DataFrame({"sidx": [0, 1, 2, 3], "tidx": [0, 1, 2, 3], "p2": [0.6, 0.6, 0.6, 0.8]})
        selected = decide_country(pred, 0.7, anchors, {"US": 0.5, "India": 0.8})
        self.assertEqual(set(selected["sidx"]), {0, 3})

    def test_country_tuning_respects_minimum_size_and_available_labels(self):
        anchors = pl.DataFrame({"sidx": [0, 1], "country": ["US", "US"]}).with_columns(pl.col("sidx").cast(pl.UInt32))
        pred = pl.DataFrame({"sidx": [0, 1], "tidx": [0, 1], "p2": [0.6, 0.4]}).with_columns(pl.col("sidx", "tidx").cast(pl.UInt32))
        truth = pred.head(1).select("sidx", "tidx")
        self.assertEqual(tune_country_thresholds(pred, truth, anchors, 0.7), {})
        cutoffs = tune_country_thresholds(pred, truth, anchors, 0.7, minimum_anchors=2)
        self.assertEqual(cutoffs, {"US": 0.6})

    def test_streamed_index_matches_eager_with_caps_across_batches(self):
        records = pl.DataFrame({"idx": range(8), "country": ["US"] * 8,
                                "core_n": ["acme motors", "beta labs"] * 4,
                                "addr_n": [f"{i} main road" for i in range(8)]})
        caps = {c: 3 for c in ("n", "c", "p", "a", "w", "q")}
        reference = build_target_index(make_keys(records), 100, caps).sort("key", "idx")
        with TemporaryDirectory() as tmp:
            actual = build_index(records, 100, Path(tmp), caps=caps, batch_rows=2).sort("key", "idx")
        self.assertTrue(reference.equals(actual))

    def test_phonetic_keys_recover_variants_without_crossing_countries(self):
        records = pl.DataFrame({"idx": [0, 1, 2], "country": ["India", "India", "France"],
                                "core_n": ["bharat", "parath", "parath"]})
        keys = make_phonetic_keys(records).sort("idx")
        self.assertEqual(keys["key"][0], keys["key"][1])
        self.assertNotEqual(keys["key"][0], keys["key"][2])

    def test_name_alignment_is_directional_blank_safe_and_pair_local(self):
        pairs = pl.DataFrame({"ntok_l": [["alpha", "motors"], ["alpha"], [], ["alpha"]],
                              "ntok_r": [["motors", "alhpa"], ["alpha", "motors"], [], []],
                              "core_l": ["alpha motors", "alpha", "", "alpha"],
                              "core_r": ["motors alhpa", "alpha motors", "", ""],
                              "addr_l": ["", "x", "", ""], "addr_r": ["", "y", "", ""]})
        result = compute_name_features(pairs, 1)
        self.assertGreater(result["token_align_min"][0], 0.9)
        self.assertEqual(result["token_align_l"][1], 1)
        self.assertLess(result["token_align_r"][1], 1)
        self.assertEqual(result["token_align_max"][2], 0)
        self.assertEqual(result["phonetic_jacc"][2], 0)
        self.assertEqual(result["noaddr_name_align"][1], 0)
        self.assertTrue(np.isfinite(result.to_numpy()).all())
        self.assertTrue(result.equals(compute_name_features(pl.concat([pairs, pairs]), 1).head(4)))
        self.assertEqual(compute_name_features(pairs.head(0), 1).height, 0)

    def test_country_loading_keeps_global_s3_ids(self):
        with TemporaryDirectory() as tmp:
            norm = Path(tmp) / "norm"
            norm.mkdir()
            for side in (1, 2, 3):
                pl.DataFrame({"idx": [0, 1, 2], "country": ["US", "India", "US"],
                              "entity_id": [f"{side}-{i}" for i in range(3)]}).with_columns(
                                  pl.col("idx").cast(pl.UInt32)).write_parquet(norm / f"train_source{side}.parquet")
            s1, tg = load_split(Path(tmp), "train", "India", ["idx", "entity_id"])
        self.assertEqual(s1["idx"].to_list(), [1])
        self.assertEqual(tg["idx"].to_list(), [1, 4])

    def test_context_index_matches_global_keyed_join(self):
        context = pl.DataFrame({"sidx": [5, 1, 3, 1], "tidx": [3, 1, 2, 0], "p1": [0.4, 0.5, 0.6, 0.7]})
        part = pl.DataFrame({"sidx": [3, 1, 1, 4], "tidx": [2, 0, 2, 3], "f": [1, 2, 3, 4]})
        index = ContextIndex(context)
        expected = part.join(context, on=["sidx", "tidx"], how="inner", maintain_order="left")
        self.assertTrue(index.attach(part).equals(expected))
        self.assertTrue(index.attach(part.head(0)).is_empty())

    def test_batched_matrix_preserves_negative_weights_and_trains_on_cpu(self):
        frame = pl.DataFrame({"f": [0.0, 0.1, 0.9, 1.0] * 20,
                              "label": [0, 0, 1, 1] * 20, "w": [2.0, 2.0, 1.0, 1.0] * 20})
        iterator = FrameIter(frame, ["f"], 7)
        batches = []
        while iterator.next(lambda **kwargs: batches.append(kwargs)):
            pass
        np.testing.assert_array_equal(np.concatenate([b["weight"] for b in batches]), frame["w"].to_numpy())
        model = fit(frame, ["f"], frame, 5,
                    {"objective": "binary:logistic", "tree_method": "hist", "max_bin": 32,
                     "max_depth": 2, "nthread": 2, "device": "cpu"})
        pred = predict_frame(model, frame, ["f"], 9)
        self.assertGreater(pred[2], pred[0])


if __name__ == "__main__":
    unittest.main()
