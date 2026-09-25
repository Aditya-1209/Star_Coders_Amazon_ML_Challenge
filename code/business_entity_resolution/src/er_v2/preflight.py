"""Fail early on a PC without working CUDA; invoked by the desktop launcher."""
from __future__ import annotations

import json
import platform
import shutil
import subprocess

import numpy as np
import polars as pl
import xgboost as xgb

from .matrix import quantile_matrix


def main() -> None:
    print(f"Python {platform.python_version()}, XGBoost {xgb.__version__}", flush=True)
    if not xgb.build_info().get("USE_CUDA", False):
        raise SystemExit("This XGBoost installation has no CUDA support. Install requirements_v2.txt on the PC.")
    executable = shutil.which("nvidia-smi")
    if executable:
        subprocess.run([executable, "--query-gpu=name,memory.total,memory.free,driver_version",
                        "--format=csv"], check=True)
    # This tiny probe runs only when the teammate invokes the launcher. It
    # verifies both the new iterator and actual GPU execution, not just imports.
    frame = pl.DataFrame({"x": np.arange(64, dtype=np.float32), "label": [0, 1] * 32})
    params = {"device": "cuda", "tree_method": "hist", "objective": "binary:logistic",
              "max_depth": 2, "max_bin": 256, "nthread": 2}
    matrix = quantile_matrix(frame, ["x"], 16, params)
    model = xgb.train(params, matrix, num_boost_round=1)
    device = json.loads(model.save_config())["learner"]["generic_param"]["device"]
    if not device.startswith("cuda"):
        raise SystemExit(f"XGBoost fell back to {device}. Update the NVIDIA driver before running this profile.")
    print(f"CUDA preflight passed on {device}. No challenge data was used.", flush=True)


if __name__ == "__main__":
    main()
