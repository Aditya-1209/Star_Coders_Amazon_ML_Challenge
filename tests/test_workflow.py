"""Exercise the teammate's documented entry point in an isolated tiny dataset."""

from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
sys.path.insert(0, str(ROOT / "scripts"))

from er_baseline.data import SOURCE_HEADER, write_tsv
from er_baseline.submission import package_submission
from extract_dataset import FILES, extract_dataset


class WorkflowTests(unittest.TestCase):
    def test_saved_model_to_validated_self_contained_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "dataset"
            write_tsv(data / "test/test_source1.tsv", SOURCE_HEADER,
                      [["S1-1", "Boulangerie Martin", "12 Rue Hugo Paris", "France"],
                       ["S1-2", "Nothing", "", "Unknown"]])
            write_tsv(data / "test/test_source2.tsv", SOURCE_HEADER,
                      [["S2-1", "Boulangerie Martin", "12 Rue Hugo Paris", "France"]])
            write_tsv(data / "test/test_source3.tsv", SOURCE_HEADER,
                      [["S3-1", "Another Business", "Elsewhere", "India"]])
            command = [sys.executable, str(ROOT / "scripts/run_full_inference.py"),
                       "--dataset", str(data), "--work", str(root / "work"),
                       "--output", str(root / "output"), "--submission", str(root / "submission.zip"),
                       "--workers", "2", "--chunk-size", "1"]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=90)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            status = json.loads((root / "work/full_inference_status.json").read_text())
            self.assertEqual(status["stage"], "complete")
            self.assertEqual(status["rows"], 2)
            with zipfile.ZipFile(root / "submission.zip") as zipped:
                self.assertIsNone(zipped.testzip())
                names = zipped.namelist()
                self.assertIn("output/matching_results.tsv", names)
                self.assertIn("output/candidate_pairs.tsv", names)
                self.assertIn("code/business_entity_resolution/model/model.cbm", names)
                self.assertFalse(any("sqlite" in name or "parts/" in name or ".venv" in name for name in names))
                doc = zipped.read("Documentation_template.md").decode()
                self.assertNotIn("<!-- FULL_TEST_RESULTS -->", doc)
                self.assertIn("Source 1 rows in each output: **2**", doc)
                self.assertIn("Advika Raj", doc)
                extracted = root / "unpacked"
                zipped.extractall(extracted)
            package = extracted / "code/business_entity_resolution"
            env = dict(os.environ, PYTHONPATH=str(package / "src"))
            reproduced = subprocess.run([
                sys.executable, "-m", "er_baseline", "predict-parallel", "--anchors", str(data / "test/test_source1.tsv"),
                "--index", str(root / "work/test_search.sqlite"), "--model", str(package / "model"),
                "--output", str(root / "reproduced"), "--workers", "2", "--chunk-size", "1",
            ], cwd=package, env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(reproduced.returncode, 0, reproduced.stdout + reproduced.stderr)
            for name in ("matching_results.tsv", "candidate_pairs.tsv"):
                self.assertEqual((root / "output" / name).read_bytes(), (root / "reproduced" / name).read_bytes())
            (root / "output/matching_results.tsv").write_text("corrupt\n")
            with self.assertRaisesRegex(ValueError, "Validated output has changed"):
                package_submission(root / "output", ROOT / "models/baseline_full_corpus",
                                   ROOT / "code/business_entity_resolution", ROOT / "Documentation_template.md",
                                   root / "should_not_exist.zip")
            self.assertFalse((root / "should_not_exist.zip").exists())

    def test_extract_ignores_unrecognized_paths_and_verifies_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with zipfile.ZipFile(root / "data.zip", "w") as zipped:
                for relative in FILES:
                    zipped.writestr("student_resource/dataset/" + relative, "example data\n")
                zipped.writestr("../../escape.txt", "must not be extracted")
            extract_dataset(root / "data.zip", root / "dataset")
            self.assertFalse((root / "escape.txt").exists())
            self.assertEqual(len(list((root / "dataset").rglob("*.tsv"))), 7)
            extract_dataset(root / "data.zip", root / "dataset")
            (root / "dataset" / FILES[0]).write_text("changed")
            with self.assertRaises(FileExistsError):
                extract_dataset(root / "data.zip", root / "dataset")


if __name__ == "__main__":
    unittest.main()
