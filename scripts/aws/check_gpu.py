"""Tiny GPU preflight executed on AWS, never during source preparation."""
import json
import numpy as np
import xgboost as xgb


def main():
    x = np.array([[0, 1], [1, 0], [0, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.float32)
    model = xgb.train({"objective": "binary:logistic", "tree_method": "hist", "device": "cuda",
                       "max_depth": 1, "nthread": 2}, xgb.DMatrix(x, label=y), num_boost_round=2)
    device = json.loads(model.save_config())["learner"]["generic_param"]["device"]
    if not device.startswith("cuda"):
        raise RuntimeError(f"Expected CUDA, got {device}; stop before spending hours on a CPU fallback")
    print(f"GPU preflight passed: XGBoost {xgb.__version__}, {device}", flush=True)


if __name__ == "__main__":
    main()
