from pathlib import Path
import tempfile
import unittest

from er_baseline.data import SOURCE_HEADER, write_tsv
from er_baseline.validate import validate_submission


class SubmissionValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        write_tsv(self.root / "anchors.tsv", SOURCE_HEADER,
                  [["S1-1", "Cafe", "Paris", "France"], ["S1-2", "Singleton", "", "India"]])
        write_tsv(self.root / "targets.tsv", SOURCE_HEADER,
                  [["S2-1", "Cafe", "Paris", "France"], ["S3-1", "Cafe", "Paris", "France"]])
        self.matching = "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\nS1-2\t\n"
        self.candidates = "source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1,S3-1\nS1-2\t\n"
        self.write_outputs()

    def tearDown(self):
        self.temporary.cleanup()

    def write_outputs(self, matching=None, candidates=None):
        (self.root / "matching_results.tsv").write_text(self.matching if matching is None else matching)
        (self.root / "candidate_pairs.tsv").write_text(self.candidates if candidates is None else candidates)

    def validate(self):
        return validate_submission(self.root / "anchors.tsv", [self.root / "targets.tsv"], self.root)

    def test_valid_outputs_with_empty_rows(self):
        result = self.validate()
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["empty_matches"], 1)
        self.assertEqual(result["candidate_pairs"], 2)

    def test_rejects_invalid_submission_variants(self):
        variants = [
            (self.matching.replace("S2-1", "S2-missing"), self.candidates),
            (self.matching, self.candidates.replace("S2-1,S3-1", "S3-1")),
            (self.matching, self.candidates.replace("S2-1,S3-1", "S2-1,S2-1")),
            (self.matching.replace("S1-2", "S1-1"), self.candidates),
            (self.matching.rsplit("S1-2", 1)[0], self.candidates),
            (self.matching + "S1-3\t\n", self.candidates),
            (self.matching.replace("matched_entity_ids", "wrong_header"), self.candidates),
        ]
        for matched, candidate in variants:
            with self.subTest(matching=matched, candidates=candidate):
                self.write_outputs(matched, candidate)
                with self.assertRaises(ValueError):
                    self.validate()


if __name__ == "__main__":
    unittest.main()
