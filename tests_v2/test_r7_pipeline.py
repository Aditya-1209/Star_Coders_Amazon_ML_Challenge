"""Tiny synthetic end-to-end test, including training, graph building and TSVs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import polars as pl


class R7PipelineTest(unittest.TestCase):
    def test_cpu_pipeline_with_enhanced_features_and_country_partitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work, dataset, model = root / "work", root / "data", root / "model"
            (work / "norm").mkdir(parents=True)
            words = ["bharat", "parath", "crimson", "cascade", "quantum", "orchard"]
            for split in ("train", "test"):
                (dataset / split).mkdir(parents=True)
                for side in (1, 2, 3):
                    n = 240 if side == 1 else 300
                    frame = pl.DataFrame({
                        "entity_id": [f"S{side}-{i}" for i in range(n)],
                        "business_name": [f"{words[i % len(words)]} business {i}" for i in range(n)],
                        "business_address": [f"{i} main street" if side == 1 or i % 4 else "" for i in range(n)],
                        "country": ["India" if i % 2 else "US" for i in range(n)]})
                    frame.write_csv(dataset / split / f"{split}_source{side}.tsv", separator="\t")
                    frame = frame.with_row_index("idx").with_columns(
                        name_n=pl.col("business_name"), core_n=pl.col("business_name"), addr_n=pl.col("business_address"))
                    frame.write_parquet(work / "norm" / f"{split}_source{side}.parquet")
            pl.DataFrame({"source1_entity_id": [f"S1-{i}" for i in range(240)],
                          "matched_entity_ids": [f"S2-{i},S3-{i}" if i % 10 else "" for i in range(240)]}).write_csv(
                              dataset / "train" / "train_ground_truth.tsv", separator="\t")
            env = {**os.environ, "POLARS_MAX_THREADS": "2", "OMP_NUM_THREADS": "2",
                   "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution/src"),
                   "PYTHONWARNINGS": "ignore::DeprecationWarning"}
            def run(module, *arguments):
                result = subprocess.run([sys.executable, "-m", "er_v2." + module, "--work", str(work),
                                         *map(str, arguments)], env=env, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr[-4000:])
            for split in ("train", "test"):
                run("run_block", "--split", split, "--top-k", 8, "--name-k", 2,
                    "--address-k", 2, "--rescue-k", 2, "--phonetic-k", 2)
                before = pl.read_parquet(work / f"cands_{split}.parquet").sort("sidx", "tidx")
                run("run_block", "--split", split, "--top-k", 8, "--name-k", 2,
                    "--address-k", 2, "--rescue-k", 2, "--phonetic-k", 2, "--augment-phonetic")
                after = pl.read_parquet(work / f"cands_{split}.parquet").sort("sidx", "tidx")
                self.assertTrue(before.select("sidx", "tidx", "rescue").equals(after.select("sidx", "tidx", "rescue")))
                run("run_features", "--split", split, "--dataset", dataset,
                    "--enhanced", "--workers", 2, "--shard-pairs", 700)
            runtime = ("--model-dir", model, "--device", "cpu", "--threads", 2, "--batch-rows", 200)
            train = ("--dataset", dataset, "--rounds", 16, "--feature-profile", "enhanced", "--compare-baseline", "--country-thresholds")
            run("train", *runtime, *train)
            run("stage3", *runtime, *train, "--split", "train", "--support-anchors", 25)
            out = root / "output"
            run("predict", *runtime, "--output", out)
            run("stage3", *runtime, "--split", "test", "--support-anchors", 25, "--output", out)
            meta = json.loads((model / "metrics.json").read_text())
            graph = json.loads((model / "stage3_metrics.json").read_text())
            self.assertEqual(len(meta["feature_trials"]), 2)
            self.assertEqual(len(graph["feature_trials"]), 2)
            self.assertEqual(graph["selection_fold"], 3)
            for name in ("candidate_pairs.tsv", "matching_results.tsv"):
                rows = pl.read_csv(out / name, separator="\t", infer_schema=False)
                self.assertEqual(rows.height, 240)
                self.assertEqual(rows["source1_entity_id"].n_unique(), 240)
            validator = Path(__file__).resolve().parents[1] / "student_resource/utils/validate_submission.py"
            if validator.exists():
                result = subprocess.run([sys.executable, str(validator), "--matching", str(out / "matching_results.tsv"),
                    "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", str(dataset / "test"), "--check-ids"],
                    capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
