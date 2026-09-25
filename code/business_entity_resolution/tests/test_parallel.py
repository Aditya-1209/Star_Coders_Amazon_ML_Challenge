import json
from pathlib import Path
import tempfile
import unittest

from catboost import CatBoostClassifier
import numpy as np

from er_baseline.data import Record, SOURCE_HEADER, write_tsv
from er_baseline.model import predict
from er_baseline.parallel import FILES, predict_parallel
from er_baseline.retrieval import build_index
from er_baseline.text import FEATURE_NAMES


class ParallelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        targets = [Record("S2-1", "Boulangerie Martin", "12 Rue Hugo Paris", "France"),
                   Record("S3-2", "Al Builders", "12 Ring Rd Nagpur", "India")]
        self.anchors = [Record("S1-1", "Boulangerie Martin", "12 Rue Hugo Paris", "France"),
                        Record("S1-2", "Al Builders", "12 Ring Road Nagpur", "India"),
                        Record("S1-3", "Nothing Similar", "", "New Country"),
                        Record("S1-4", "Martin", "12 Rue Hugo", "France"),
                        Record("S1-5", "Al Builders", "", "India")]
        write_tsv(self.root / "targets.tsv", SOURCE_HEADER, (r.values() for r in targets))
        write_tsv(self.root / "anchors.tsv", SOURCE_HEADER, (r.values() for r in self.anchors))
        build_index([self.root / "targets.tsv"], self.root / "index.sqlite")
        model_dir = self.root / "model"
        model_dir.mkdir()
        model = CatBoostClassifier(iterations=3, depth=2, thread_count=1, allow_writing_files=False, verbose=False)
        x = np.vstack([np.zeros((8, len(FEATURE_NAMES))), np.ones((8, len(FEATURE_NAMES)))])
        model.fit(x, [0] * 8 + [1] * 8)
        model.save_model(str(model_dir / "model.cbm"))
        (model_dir / "config.json").write_text(json.dumps({"threshold": 0.5, "feature_names": FEATURE_NAMES, "per_channel": 2, "posting_budget": 3}))

    def tearDown(self):
        self.temporary.cleanup()

    def run_parallel(self, output, **options):
        return predict_parallel(self.root / "anchors.tsv", self.root / "index.sqlite", self.root / "model", self.root / output, workers=2, chunk_size=2, **options)

    def test_resume_is_identical_to_serial_and_keeps_empty_rows(self):
        partial = self.run_parallel("parallel", max_chunks=1)
        self.assertEqual(partial["status"], "partial")
        self.assertEqual(partial["completed_rows"], 2)
        self.assertFalse((self.root / "parallel/matching_results.tsv").exists())
        part = self.root / "parallel/parts/000000.matching_results.tsv"
        original_mtime = part.stat().st_mtime_ns
        completed = self.run_parallel("parallel", resume=True)
        self.assertEqual(completed["status"], "complete")
        self.assertEqual(completed["resumed_rows"], 2)
        self.assertEqual(part.stat().st_mtime_ns, original_mtime)
        predict(self.root / "anchors.tsv", self.root / "index.sqlite", self.root / "model", self.root / "serial")
        for name in FILES:
            self.assertEqual((self.root / "parallel" / name).read_bytes(), (self.root / "serial" / name).read_bytes())
            self.assertIn("S1-3\t\n", (self.root / "parallel" / name).read_text())

    def test_resume_rejects_corrupt_checkpoint(self):
        self.run_parallel("parallel", max_chunks=1)
        (self.root / "parallel/parts/000000.candidate_pairs.tsv").write_text("corrupt\n")
        with self.assertRaisesRegex(ValueError, "Checkpoint corruption"):
            self.run_parallel("parallel", resume=True)

    def test_resume_rejects_changed_input(self):
        self.run_parallel("parallel", max_chunks=1)
        write_tsv(self.root / "anchors.tsv", SOURCE_HEADER, (r.values() for r in reversed(self.anchors)))
        with self.assertRaisesRegex(ValueError, "Resume rejected"):
            self.run_parallel("parallel", resume=True)


if __name__ == "__main__":
    unittest.main()
