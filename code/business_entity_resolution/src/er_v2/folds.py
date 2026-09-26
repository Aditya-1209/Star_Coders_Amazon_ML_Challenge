"""Stable fold identities without importing either neural or tree runtimes."""
import polars as pl

N_FOLDS = 10

def fold_expr() -> pl.Expr:
    return (pl.col("sidx").hash(seed=11) % N_FOLDS).cast(pl.Int8).alias("fold")
