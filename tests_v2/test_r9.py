"""R9 protocol tests. Written for the AWS checks stage; no full data needed."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl

from er_v2.graph import anchors_of
from er_v2.metrics import macro_f05
from er_v2.r9 import business_weights, crossfit_inputs, mixed_predictions, paired_gate, per_business, tune_half
from er_v2.train import fold_expr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_r9


def pairs(sidx, tidx, **columns):
    return pl.DataFrame({"sidx": sidx, "tidx": tidx, **columns}).with_columns(pl.col("sidx", "tidx").cast(pl.UInt32))


class R9Tests(unittest.TestCase):
    def test_business_loss_weight_is_equal_per_business_and_mean_one(self):
        frame = pairs([0, 1, 1, 1], [0, 1, 2, 3], label=[1, 0, 0, 1])
        weighted = business_weights(frame)
        np.testing.assert_allclose(weighted.group_by("sidx").agg(pl.col("w").sum())["w"], [2, 2])
        self.assertAlmostEqual(weighted["w"].mean(), 1.0)

    def test_model_crossfit_never_scores_its_own_training_fold(self):
        ids = pl.DataFrame({"sidx": range(1000)}).with_columns(pl.col("sidx").cast(pl.UInt32)).with_columns(fold_expr())
        selected = [ids.filter(pl.col("fold") == f)["sidx"][0] for f in (6, 7, 3, 4)]
        frame = pairs(selected, [0, 1, 2, 3], p1=[0.2] * 4, direct=[1] * 4,
                      sup_both_max=[0.] * 4, sup_addr_valid=[1] * 4, sup_name_tset=[0.] * 4)
        calls = []
        def predict(model, data, features, batch_rows):
            calls.append((model, data.select(fold_expr())["fold"].to_list()))
            return np.full(len(data), model / 10, dtype=np.float32)
        with patch("er_v2.r9.predict_frame", side_effect=predict):
            out = crossfit_inputs(frame, {6: 6, 7: 7}, ["p1"], 2, True)
        np.testing.assert_allclose(out["p2"], [0.7, 0.6, 0.65, 0.65])
        for model, scored_folds in calls:
            self.assertNotIn(model, scored_folds)
        # Test row indices can hash to training folds, but every test row must use both models.
        with patch("er_v2.r9.predict_frame", side_effect=predict):
            out = crossfit_inputs(frame, {6: 6, 7: 7}, ["p1"], 2, False)
        np.testing.assert_allclose(out["p2"], [0.65] * 4)

    def test_blend_is_keyed_preserves_union_and_handles_missing_evidence(self):
        new = pairs([1, 0], [2, 0], score=[0.8, 0.6])
        base = pairs([0, 2], [0, 3], score=[1., 0.4])
        mixed = mixed_predictions(new, base, 0.75).sort("sidx")
        self.assertEqual(mixed["tidx"].to_list(), [0, 2, 3])
        np.testing.assert_allclose(mixed["score"], [0.7, 0.6, 0.1])

    def test_macro_delta_includes_empty_businesses_and_duplicate_pairs(self):
        truth = pairs([0, 0, 1], [0, 1, 2])
        pred = pairs([0, 0, 1], [0, 0, 2])
        anchors = pl.Series([0, 1, 2, 3], dtype=pl.UInt32)
        actual = per_business(pred, truth, anchors)["f"].mean()
        self.assertAlmostEqual(actual, macro_f05(pred, truth, anchors)["macro_f05"])
        same = paired_gate(pred, pred, truth, anchors, 0.0005)
        self.assertFalse(same["passed"])
        self.assertEqual(same["gain"], 0.)
        bad = pairs([2, 3], [8, 9])
        gate = paired_gate(bad, pred, truth, anchors, 0.0005)
        self.assertFalse(gate["passed"])

    def test_expansion_uses_only_confident_exclusive_anchors(self):
        frame = pairs([0, 1, 2], [7, 7, 8], p2=[0.95, 0.9, 0.7])
        self.assertEqual(anchors_of(frame, threshold=0.8)["sidx"].to_list(), [0])

    def test_tuning_halves_partition_businesses_deterministically(self):
        ids = pl.DataFrame({"sidx": range(2000)}).with_columns(pl.col("sidx").cast(pl.UInt32))
        tune = ids.filter(fold_expr() == 3)
        a, b = tune.filter(tune_half() == 0), tune.filter(tune_half() == 1)
        self.assertTrue(len(a) and len(b))
        self.assertEqual(len(a) + len(b), len(tune))
        self.assertFalse(set(a["sidx"]) & set(b["sidx"]))
        self.assertTrue(a.equals(tune.filter(tune_half() == 0)))

    def test_country_proxy_excludes_supervision_at_every_level(self):
        args = run_r9.parser().parse_args(["--exclude-country", "India"])
        args.work = Path("/tmp/r9-protocol-unused")
        plan = run_r9.commands(args)
        for step in ("translit", "train", "stage3_train", "crossfit", "fit", "select"):
            command = plan[step][0]
            self.assertIn("--exclude-country", command)
            self.assertEqual(command[command.index("--exclude-country") + 1], "India")
        command = plan["translit"][0]
        start = command.index("--learn-folds")
        self.assertEqual(command[start + 1:start + 5], ["0", "1", "8", "9"])
        self.assertIn(args.work / "base/stage3_train_noIndia.parquet", plan["stage3_train"][1])
        self.assertLess(run_r9.STEPS.index("select"), run_r9.STEPS.index("evaluate"))


if __name__ == "__main__":
    unittest.main()
