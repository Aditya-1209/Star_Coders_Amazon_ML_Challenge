"""Construct quantized training matrices without a full dense float32 copy."""
from __future__ import annotations

import polars as pl
import xgboost as xgb


class FrameIter(xgb.DataIter):
    """Resettable dense batches; source frame and quantized matrix remain in RAM."""

    def __init__(self, frame: pl.DataFrame, features: list[str], batch_rows: int):
        if batch_rows < 1:
            raise ValueError("matrix batch_rows must be positive")
        self.frame = frame
        self.features = features
        self.batch_rows = batch_rows
        self.position = 0
        self._batch_features = None
        self._batch_labels = None
        super().__init__(release_data=True)

    def reset(self) -> None:
        self.position = 0
        self._batch_features = None
        self._batch_labels = None

    def next(self, input_data) -> bool:
        if self.position >= len(self.frame):
            return False
        batch = self.frame.slice(self.position, self.batch_rows)
        self.position += len(batch)
        self._batch_features = batch.select(pl.col(self.features).cast(pl.Float32)).to_numpy()
        self._batch_labels = batch["label"].to_numpy()
        input_data(data=self._batch_features, label=self._batch_labels, feature_names=self.features)
        return True


def quantile_matrix(frame: pl.DataFrame, features: list[str], batch_rows: int,
                    params: dict, reference=None):
    iterator = FrameIter(frame, features, batch_rows)
    return xgb.QuantileDMatrix(iterator, ref=reference, max_bin=params.get("max_bin", 256),
                               nthread=params.get("nthread", 12))
