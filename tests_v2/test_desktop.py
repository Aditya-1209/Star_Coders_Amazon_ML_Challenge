"""Desktop resource-path checks. Written for the PC; not run on the Mac."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from er_v2.block import build_target_index, generate, make_keys
from er_v2.features import add_block_context, compute, pair_frame, record_frames
from er_v2.matrix import FrameIter, quantile_matrix
from er_v2.prepare import _norm_chunk, normalize_frame
from er_v2.run_block import load_split, split_countries, target_count
from er_v2.stage3 import build
from er_v2.train import ContextBatches, context_features
from scripts.run_r4_desktop import affected_steps, build_steps, output_snapshot, verify_completed


def normalized(ids, countries, names):
    return pl.DataFrame({"idx": ids, "entity_id": [f"id-{i}" for i in ids], "country": countries,
                         "business_name": names, "name_n": names, "core_n": names,
                         "business_address": ["11 river road"] * len(ids),
                         "addr_n": ["11 river rd"] * len(ids)}).with_columns(pl.col("idx").cast(pl.UInt32))


class DesktopTests(unittest.TestCase):
    def test_context_slices_match_keyed_join_and_preserve_tie_breaks(self):
        scores = pl.DataFrame({"sidx": [2, 0, 1, 0, 2, 1], "tidx": [7, 8, 7, 7, 9, 9],
                               "p1": [0.8, 0.5, 0.8, 0.8, 0.0, 0.3]}).with_columns(pl.col("sidx", "tidx").cast(pl.UInt32))
        context = context_features(scores)
        assert_frame_equal(context.select("sidx", "tidx"), scores.select("sidx", "tidx"))
        claimants = context.filter(pl.col("tidx") == 7).sort("sidx")
        self.assertEqual(claimants["t_prank"].to_list(), [1, 2, 3])
        source_zero = context.filter(pl.col("sidx") == 0).sort("tidx")
        self.assertEqual(source_zero["s_rank"].to_list(), [1, 2])
        features = scores.select("sidx", "tidx").with_columns(f=pl.Series(range(6)))
        expected = features.join(context.filter(pl.col("p1") >= 0.2), on=["sidx", "tidx"], maintain_order="left")
        batches = ContextBatches(context, 0.2)
        actual = pl.concat([batches.attach(part) for part in features.iter_slices(2)])
        batches.finish()
        assert_frame_equal(actual, expected)
        incomplete = ContextBatches(context, 0.2)
        with self.assertRaisesRegex(ValueError, "ended before"):
            incomplete.finish()
        with self.assertRaisesRegex(ValueError, "row order differs"):
            incomplete.attach(features.reverse())

    def test_matrix_batches_preserve_rows_labels_feature_order_and_reset(self):
        frame = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50], "label": [0, 1, 0, 1, 0]})
        iterator = FrameIter(frame, ["b", "a"], 2)
        for _ in range(2):
            batches = []
            def receive(**kwargs):
                batches.append((kwargs["data"].copy(), kwargs["label"].copy(), kwargs["feature_names"]))
            while iterator.next(receive):
                pass
            self.assertEqual([len(x) for x, _, _ in batches], [2, 2, 1])
            np.testing.assert_array_equal(np.concatenate([x for x, _, _ in batches]), [[10, 1], [20, 2], [30, 3], [40, 4], [50, 5]])
            np.testing.assert_array_equal(np.concatenate([y for _, y, _ in batches]), frame["label"].to_numpy())
            self.assertEqual(batches[0][0].dtype, np.float32)
            self.assertEqual(batches[0][2], ["b", "a"])
            iterator.reset()

    def test_validation_quantization_uses_training_reference_and_same_bins(self):
        frame = pl.DataFrame({"f": [1.0, 2.0], "label": [0, 1]})
        reference = object()
        with patch("er_v2.matrix.xgb.QuantileDMatrix") as constructor:
            quantile_matrix(frame, ["f"], 1, {"max_bin": 128, "nthread": 3}, reference)
        self.assertIs(constructor.call_args.kwargs["ref"], reference)
        self.assertEqual(constructor.call_args.kwargs["max_bin"], 128)
        self.assertEqual(constructor.call_args.kwargs["nthread"], 3)

    def test_normalization_queues_only_one_window_and_keeps_record_order(self):
        class Pool:
            windows = []

            def imap(self, fn, jobs):
                jobs = list(jobs)
                self.windows.append(sum(len(names) for names, _ in jobs))
                return map(fn, jobs)

        raw = pl.DataFrame({"business_name": [f"Acme {i}" for i in range(7)],
                            "business_address": [f"{i} Main Street" for i in range(7)]})
        pool = Pool()
        actual = normalize_frame(raw, pool, chunk=2, buffer_rows=3)
        full, core, addr = _norm_chunk((raw["business_name"].to_list(), raw["business_address"].to_list()))
        expected = raw.with_columns(pl.Series("name_n", full), pl.Series("core_n", core), pl.Series("addr_n", addr))
        assert_frame_equal(actual, expected)
        self.assertEqual(pool.windows, [3, 3, 1])

    def test_country_partition_preserves_global_ids_idf_candidates_and_graph_features(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "norm").mkdir()
            normalized([0, 1], ["India", "US"], ["alpha", "beta"]).write_parquet(work / "norm/test_source1.parquet")
            normalized([0, 1], ["US", "India"], ["beta", "alpha"]).write_parquet(work / "norm/test_source2.parquet")
            normalized([0, 1], ["India", "US"], ["alpha", "beta"]).write_parquet(work / "norm/test_source3.parquet")
            left, right = load_split(work, "test")
            total = target_count(work, "test")
            self.assertEqual(total, 4)
            _, india = load_split(work, "test", "India")
            self.assertEqual(sorted(india["idx"]), [1, 2])
            whole = generate(make_keys(left), build_target_index(make_keys(right), total), top_k=64,
                              chunk=1, name_k=16, address_k=8, verbose=False)
            pieces = []
            for country in split_countries(work, "test"):
                source, target = load_split(work, "test", country)
                pieces.append(generate(make_keys(source), build_target_index(make_keys(target), total),
                                       top_k=64, name_k=16, address_k=8, verbose=False))
            assert_frame_equal(whole.sort("sidx", "tidx"), pl.concat(pieces).sort("sidx", "tidx"), check_exact=False)
            left_records, right_records = record_frames(left, right)
            features = compute(pair_frame(add_block_context(whole), left_records, right_records, 2), workers=1)
            # Only India needs an entirely new hop feature row; US keeps both
            # direct records. This also checks cross-country schema consistency.
            known = whole.filter(~((pl.col("sidx") == 0) & (pl.col("tidx") == 2)))
            features = features.join(known.select("sidx", "tidx"), on=["sidx", "tidx"], how="semi")
            folder = work / "feats_test"
            folder.mkdir()
            features.write_parquet(folder / "part_000.parquet")
            stage2 = known.select("sidx", "tidx").with_columns(p1=pl.lit(0.9, pl.Float32), p2=pl.lit(0.9, pl.Float32))
            options = dict(batch_rows=2, support_anchors=1, workers=1, block_chunk=1)
            complete = build("test", work, stage2, folder, lambda _: None, **options)
            partitioned = build("test", work, stage2, folder, lambda _: None, country_partition=True, **options)
            assert_frame_equal(complete.sort("sidx", "tidx"), partitioned.sort("sidx", "tidx"), check_exact=False)

    def test_desktop_commands_keep_accuracy_settings_and_use_target_resources(self):
        import json
        root = Path(__file__).resolve().parents[1]
        profile = json.loads((root / "configs/desktop_13900k_3060.json").read_text())
        steps = dict(build_steps(profile, Path("work"), Path("out"), Path("data"), Path("validator.py")))
        self.assertIn("cuda", steps["train"])
        self.assertIn("cpu", steps["predict"])
        self.assertNotIn("--no-extra-stage1-folds", steps["train"])
        for split in ("train", "test"):
            self.assertIn("--country-partition", steps["block_" + split])
            self.assertIn("--country-partition", steps["features_" + split])
            command = steps["block_" + split]
            self.assertEqual(command[command.index("--top-k") + 1], "64")
        self.assertEqual(steps["train"][steps["train"].index("--matrix-batch-rows") + 1], "100000")

    def test_resume_rejects_changed_settings_or_missing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "models").mkdir()
            artifact = work / "models/translit.json"
            artifact.write_text("{}")
            command = ["python", "translit"]
            record = {"command": command, "outputs": output_snapshot("translit", work, work / "out")}
            verify_completed("translit", command, record, work, work / "out")
            with self.assertRaises(SystemExit):
                verify_completed("translit", command + ["--changed"], record, work, work / "out")
            artifact.unlink()
            with self.assertRaises(SystemExit):
                verify_completed("translit", command, record, work, work / "out")

    def test_retraining_invalidates_predictions_but_keeps_test_preprocessing(self):
        self.assertEqual(affected_steps("train"), {"train", "stage3_train", "predict", "stage3_predict", "validate"})
        self.assertIn("features_test", affected_steps("translit"))
        self.assertNotIn("train", affected_steps("features_test"))


if __name__ == "__main__":
    unittest.main()
