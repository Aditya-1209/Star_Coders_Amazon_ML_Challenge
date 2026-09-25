"""Build country indexes without materializing every exploded key at once."""
from pathlib import Path
from tempfile import TemporaryDirectory
import math
import polars as pl

from .block import CAPS, make_keys


def build_index(records: pl.DataFrame, n_targets: int, work: Path,
                maker=make_keys, caps=CAPS, batch_rows: int = 50_000) -> pl.DataFrame:
    if not len(records) or n_targets < 1:
        raise ValueError("Cannot index an empty target corpus")
    work.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="keys-", dir=work) as tmp:
        for i, part in enumerate(records.iter_slices(batch_rows)):
            maker(part).write_parquet(Path(tmp) / f"{i:05d}.parquet")
        keys = pl.scan_parquet(str(Path(tmp) / "*.parquet"))
        counts = keys.group_by("key", "kind").agg(df=pl.len())
        counts = counts.filter(pl.col("df") <= pl.col("kind").replace_strict(caps, return_dtype=pl.UInt32))
        weights = counts.select("key", w=(math.log(n_targets) - pl.col("df").cast(pl.Float64).log()).cast(pl.Float32))
        # Materialize the much smaller counts before joining; keep key batches
        # on disk until the streaming join has finished.
        weights = weights.collect(engine="streaming")
        return keys.join(weights.lazy(), on="key").select("key", "idx", "w", "kind").collect(engine="streaming")
