"""Stage 3: two-hop expansion and record-to-record support features.

Every Source 1 business owns several Source 2/3 records, and those records
resemble each other (shared address, name variants, website form). After
stage 2, the confident matches of a business ("anchors", p2 >= ANCHOR) are
used as extra queries:

1. Expansion: each anchor is run through the same weighted key blocking as a
   Source 1 query, against all Source 2/3 records; its top HOP_K neighbours
   become candidates of the anchor's business ("two-hop" candidates).
2. Support: for every candidate of a business, the similarity of the candidate
   to that business's anchors (best name/address match, anchor-weighted).

The stage-3 model re-scores the union of pruned direct candidates and two-hop
candidates using pair features + stage-1/2 scores + support features.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from .block import generate

ANCHOR = 0.5
HOP_K = 10
# A two-hop candidate is kept only if it resembles one of the business's anchors on
# both name and address (min of the two token-set scores). Measured on fold 3:
# 19.7 -> 5.6 candidates per Source 1 at unchanged macro F0.5 (0.9631 -> 0.9630).
HOP_MIN_SUPPORT = 50


def anchors_of(stage2: pl.DataFrame) -> pl.DataFrame:
    """Confident matches after exclusivity: (sidx, a, pa)."""
    a = stage2.filter(pl.col("p2") >= ANCHOR)
    a = a.filter(pl.col("p2") == pl.col("p2").max().over("tidx")).unique("tidx", keep="first")
    return a.select("sidx", a="tidx", pa="p2")


def expand(anchors: pl.DataFrame, tkeys: pl.DataFrame, tindex: pl.DataFrame) -> pl.DataFrame:
    """Two-hop candidates: (sidx, tidx, hop_score, hop_rank, hop_n)."""
    ak = tkeys.join(anchors.select(idx="a").unique(), on="idx", how="semi")
    nb = generate(ak, tindex, top_k=HOP_K + 1, verbose=False)
    nb = nb.rename({"sidx": "a", "tidx": "t"}).filter(pl.col("a") != pl.col("t"))
    x = anchors.join(nb, on="a")
    return x.group_by("sidx", "t").agg(
        hop_score=pl.col("bscore").max(),
        hop_rank=pl.col("brank").min().cast(pl.UInt16),
        hop_n=pl.len().cast(pl.UInt16),
        hop_pa=pl.col("pa").max(),
    ).rename({"t": "tidx"})


def support_features(pairs: pl.DataFrame, anchors: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
    """Similarity of each candidate to the other anchors of its business.

    ``right`` holds per-target text columns (idx, core_r, addr_r, cc_r) as
    produced by features.record_frames.
    """
    x = pairs.select("sidx", "tidx").join(anchors, on="sidx").filter(pl.col("a") != pl.col("tidx"))
    txt = right.select("idx", "core_r", "addr_r", "cc_r")
    x = x.join(txt, left_on="tidx", right_on="idx", how="left")
    x = x.join(txt.rename({"core_r": "core_a", "addr_r": "addr_a", "cc_r": "cc_a"}),
               left_on="a", right_on="idx", how="left")
    sims = {}
    for name, (l, r, scorer) in {
        "sup_name_tset": ("core_r", "core_a", fuzz.token_set_ratio),
        "sup_name_ratio": ("core_r", "core_a", fuzz.ratio),
        "sup_cc_partial": ("cc_r", "cc_a", fuzz.partial_ratio),
        "sup_addr_tset": ("addr_r", "addr_a", fuzz.token_set_ratio),
        "sup_addr_ratio": ("addr_r", "addr_a", fuzz.ratio),
    }.items():
        sims[name] = process.cpdist(x[l].to_list(), x[r].to_list(), scorer=scorer, workers=-1,
                                    dtype=np.float32)
    x = x.select("sidx", "tidx", "pa", "addr_r", "addr_a").with_columns(**{k: pl.Series(v) for k, v in sims.items()})
    x = x.with_columns(
        sup_both=pl.min_horizontal(pl.col("sup_name_tset"), pl.col("sup_addr_tset"))
        .cast(pl.Float32),
        sup_max=pl.max_horizontal(pl.col("sup_name_tset"), pl.col("sup_addr_tset")),
        both_addr=((pl.col("addr_r") != "") & (pl.col("addr_a") != "")).cast(pl.Int8),
    )
    return x.group_by("sidx", "tidx").agg(
        n_anchor=pl.len().cast(pl.UInt16),
        sup_both_max=pl.col("sup_both").max(),
        sup_both_w=(pl.col("sup_both") * pl.col("pa")).max(),
        sup_max_max=pl.col("sup_max").max(),
        sup_name_tset=pl.col("sup_name_tset").max(),
        sup_name_ratio=pl.col("sup_name_ratio").max(),
        sup_cc_partial=pl.col("sup_cc_partial").max(),
        sup_addr_tset=pl.col("sup_addr_tset").max(),
        sup_addr_ratio=pl.col("sup_addr_ratio").max(),
        sup_addr_valid=pl.col("both_addr").max(),
        sup_n90=((pl.col("sup_max") >= 90)).sum().cast(pl.UInt16),
        sup_nboth80=((pl.col("sup_both") >= 80)).sum().cast(pl.UInt16),
        pa_max=pl.col("pa").max(),
    )
