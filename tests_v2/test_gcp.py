"""Cloud launch/export contracts. Tests never call Google APIs or train a model."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/gcp" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GCPTests(unittest.TestCase):
    def test_launch_has_gpu_runtime_limit_and_no_automatic_restart(self):
        module = load("create_vm")
        args = SimpleNamespace(project="my-r9-project", name="r9-experiment", run_id="first-run",
                               zone="us-central1-a", bucket="my-private-r9-bucket", source_sha256="a" * 64,
                               service_account="r9-runner@my-r9-project.iam.gserviceaccount.com",
                               hours=12, disk_gb=200, subnet="r9-subnet", machine_type="g2-standard-16",
                               exclude_country="none")
        module.check(args)
        cfg = module.config(args)
        self.assertEqual(cfg["threads"], 12)
        self.assertEqual(cfg["results_uri"], "gs://my-private-r9-bucket/r9/results/first-run")
        cmd = module.command(args, "/tmp/config.json")
        self.assertEqual(cmd[cmd.index("--max-run-duration") + 1], "12h")
        self.assertEqual(cmd[cmd.index("--instance-termination-action") + 1], "STOP")
        self.assertEqual(cmd[cmd.index("--provisioning-model") + 1], "STANDARD")
        self.assertIn("--no-restart-on-failure", cmd)
        self.assertNotIn("tpu", cmd)
        args.hours = 0
        with self.assertRaises(ValueError):
            module.check(args)

    def test_exports_do_not_include_dataset_features_or_partial_output(self):
        module = load("export_results")
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in ("student_resource/dataset/train/train_source1.tsv", "work/r9/graph_train.parquet",
                             "work/r9/result.json", "work/r9/models/pair.json", "work/r9/logs/train.log",
                             "output/r9/matching_results.tsv", "output/r9/matching_results.tsv.tmp"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("test")
            small = module.selected_files(root, False)
            full = module.selected_files(root, True)
            self.assertIn(Path("work/result.json"), small)
            self.assertNotIn(Path("work/models/pair.json"), small)
            self.assertIn(Path("work/models/pair.json"), full)
            self.assertIn(Path("output/matching_results.tsv"), full)
            self.assertFalse(any(p.suffix in {".parquet", ".tmp"} for p in full))
            self.assertFalse(any("student_resource" in str(p) for p in full.values()))


if __name__ == "__main__":
    unittest.main()
