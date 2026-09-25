from pathlib import Path
import tempfile
import unittest

import numpy as np

from er_baseline.data import Record, SOURCE_HEADER, read_records, write_tsv
from er_baseline.metrics import aggregate, evaluate, f05, tune_threshold
from er_baseline.prepare import group_split
from er_baseline.retrieval import Retriever, build_index
from er_baseline.text import FEATURE_NAMES, features, normalize


class MetricTests(unittest.TestCase):
    def test_official_example_and_singletons(self):
        self.assertAlmostEqual(f05({"a", "b"}, {"a", "b", "c"}), 5 / 7)
        self.assertEqual(f05([], []), 1)
        self.assertEqual(f05([], ["a"]), 0)
        self.assertEqual(f05(["a"], []), 0)

    def test_macro_includes_missed_candidates_and_singletons(self):
        result = aggregate(np.array([2, 0, 1]), np.array([1, 0, 0]), np.array([1, 0, 0]))
        self.assertAlmostEqual(result["macro_f05"], (5 / 6 + 1) / 3)
        self.assertAlmostEqual(result["pair_recall"], 1 / 3)
        self.assertEqual(result["singleton_accuracy"], 1)

    def test_threshold_uses_only_selected_entities(self):
        scores = np.array([0.8, 0.6, 0.95])
        labels = np.array([1, 0, 0])
        owners = np.array([0, 0, 1])
        truth = np.array([1, 0])
        selected = np.array([True, False])
        threshold, score, _ = tune_threshold(scores, labels, owners, truth, selected)
        self.assertTrue(0.6 < threshold <= 0.8)
        self.assertEqual(score, 1)
        holdout = evaluate(scores, labels, owners, truth, ~selected, threshold)
        self.assertEqual(holdout["macro_f05"], 0)


class DataTests(unittest.TestCase):
    def test_shared_targets_cannot_cross_split(self):
        anchors = {f"S1-{i}": Record(f"S1-{i}", "Name", "Address", "France" if i % 2 else "India") for i in range(100)}
        truth = {s: {"S2-" + s[3:]} for s in anchors}
        truth["S1-1"] = {"S2-0", "S3-5"}
        truth["S1-2"] = {"S3-5"}
        split = group_split(anchors, truth, 42)
        self.assertEqual(split["S1-0"], split["S1-1"])
        self.assertEqual(split["S1-1"], split["S1-2"])
        self.assertEqual(split, group_split(anchors, truth, 42))
        self.assertEqual(set(split.values()), {"train", "tune", "holdout"})

    def test_normalization_preserves_indic_text(self):
        self.assertEqual(normalize("École & Co."), "ecole and co")
        self.assertEqual(normalize("अल बिल्डर्स"), "अल बिल्डर्स")

    def test_blank_addresses_are_not_exact_match_evidence(self):
        a = Record("S1-1", "Acme", "", "US")
        b = Record("S2-1", "Acme", "", "US")
        values = dict(zip(FEATURE_NAMES, features(a, b, [1, 0, 2])))
        self.assertEqual(values["address_exact_nonempty"], 0)
        self.assertEqual(values["address_ratio"], 0)
        self.assertEqual(values["address_missing_left"], 1)
        self.assertEqual(len(values), len(FEATURE_NAMES))


class RetrievalTests(unittest.TestCase):
    def test_bm25_ranks_true_match_above_earlier_partial_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            targets = [Record(f"S2-{i}", f"Holy Landscaping {i}", "1 Other Street", "US") for i in range(25)]
            targets += [Record("S3-99", "Holy Ministries", "9 Woodland Drive", "US")]
            write_tsv(root / "targets.tsv", SOURCE_HEADER, (r.values() for r in targets))
            build_index([root / "targets.tsv"], root / "index.sqlite")
            r = Retriever(root / "index.sqlite", per_channel=1)
            ranked = r.db.execute('SELECT rowid,rank FROM fts_0 WHERE fts_0 MATCH ? ORDER BY rank', ('name : ("holy" OR "ministries")',)).fetchall()
            self.assertEqual(ranked[0][0], 26)
            self.assertLess(ranked[0][1], ranked[-1][1])
            result = r.query(Record("S1-1", "Holy Ministries", "9 Woodland Drive", "US"))
            self.assertEqual(result[0][0].entity_id, "S3-99")
            r.close()
            budgeted = Retriever(root / "index.sqlite", per_channel=1, posting_budget=1)
            rare = budgeted.query(Record("S1-1", "Holy Ministries", "9 Woodland Drive", "US"))
            self.assertEqual(rare[0][0].entity_id, "S3-99")
            # If all terms exceed the budget, the intersection fallback must
            # still retrieve common names rather than silently return nothing.
            common = budgeted.query(Record("S1-2", "Holy Landscaping", "", "US"))
            self.assertTrue(common)
            self.assertTrue(any(record.business_name.startswith("Holy Landscaping") for record, _ in common))
            budgeted.close()

    def test_country_generalization_fuzzy_names_and_unique_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            targets = [
                Record("S2-1", "Boulangerie Martin", "12 Rue Victor Hugo, Paris", "France"),
                Record("S3-1", "Boulangerie Martin", "12 Rue Victor Hugo", "US"),
                Record("S2-2", "अल बिल्डर्स", "12 Ring Road Nagpur", "India"),
            ]
            write_tsv(root / "targets.tsv", SOURCE_HEADER, (r.values() for r in targets))
            build_index([root / "targets.tsv"], root / "index.sqlite")
            r = Retriever(root / "index.sqlite", per_channel=3)
            found = r.query(Record("S1-1", "Boulangrie Martn", "12 rue Victor Hugo", "France"))
            self.assertEqual([target.entity_id for target, _ in found], ["S2-1"])
            self.assertEqual(len({target.entity_id for target, _ in found}), len(found))
            india = r.query(Record("S1-2", "अल बिल्डर्स", "", "India"))
            self.assertIn("S2-2", [target.entity_id for target, _ in india])
            self.assertEqual(r.query(Record("S1-3", "Unknown", "", "New Country")), [])
            r.close()


if __name__ == "__main__":
    unittest.main()
