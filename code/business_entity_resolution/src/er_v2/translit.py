"""Learn a transliteration dictionary from training ground truth.

Many Source 2/3 names are Indic-script renderings of the Latin Source 1 name
(e.g. "रेड वेंचर्स प्राइवेट लिमिटेड" -> anyascii "red vemcrs praivet limited").
For labelled pairs whose target name is non-Latin and whose token counts agree
with the Source 1 name, tokens are aligned by position; frequent, consistent
alignments become a token -> Latin token map applied during normalization.
Uses only the provided training labels.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

from .normalize import ascii_lower, fix_ocr, tokens

_LATIN = re.compile(r"^[\x00-\x7FÀ-ɏ]*$")


def raw_tokens(name: str) -> list[str]:
    return [fix_ocr(t) for t in tokens(ascii_lower(name))]


def learn(dataset: Path, min_count: int = 2, min_share: float = 0.5,
          learn_folds: list[int] | None = None, exclude_country: list[str] | None = None) -> dict[str, str]:
    d = dataset / "train"
    rd = lambda f: pl.read_csv(d / f, separator="\t", quote_char=None, infer_schema=False)
    gt = rd("train_ground_truth.tsv").with_columns(pl.col("matched_entity_ids").str.split(","))
    gt = gt.explode("matched_entity_ids").drop_nulls()
    # Exclude tuning/holdout folds 3/4 from supervised normalization.
    from .train import fold_expr
    s1 = rd("train_source1.tsv").with_row_index("sidx").with_columns(pl.col("sidx").cast(pl.UInt32))
    folds = [0, 1, 2, 5, 6, 7, 8, 9] if learn_folds is None else learn_folds
    if not folds or set(folds) - {0, 1, 2, 5, 6, 7, 8, 9}:
        raise ValueError("Supervised normalization may not use tuning/holdout folds 3/4")
    s1 = s1.filter(fold_expr().is_in(folds))
    if exclude_country:
        s1 = s1.filter(~pl.col("country").is_in(exclude_country))
    s1 = s1.select("entity_id", n1="business_name")
    tg = pl.concat([rd("train_source2.tsv"), rd("train_source3.tsv")]).select("entity_id", n2="business_name")
    tg = tg.filter(~pl.col("n2").str.contains(r"^[\x00-\x7FÀ-ɏ]*$"))
    pairs = gt.join(tg, left_on="matched_entity_ids", right_on="entity_id").join(
        s1, left_on="source1_entity_id", right_on="entity_id")
    counts: dict[str, Counter] = defaultdict(Counter)
    for n1, n2 in pairs.select("n1", "n2").iter_rows():
        a, b = raw_tokens(n1), raw_tokens(n2)
        if len(a) != len(b):
            continue
        for x, y in zip(a, b):
            if x != y:
                counts[y][x] += 1
            else:
                counts[y][y] += 1
    mapping = {}
    for src, c in counts.items():
        tgt, n = c.most_common(1)[0]
        tot = sum(c.values())
        if tgt != src and n >= min_count and n / tot >= min_share:
            mapping[src] = tgt
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="student_resource/dataset")
    ap.add_argument("--out", default="models/v2/translit.json")
    ap.add_argument("--learn-folds", nargs="+", type=int)
    ap.add_argument("--exclude-country", nargs="*", default=[])
    args = ap.parse_args()
    m = learn(Path(args.dataset), learn_folds=args.learn_folds, exclude_country=args.exclude_country)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(m, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")
    print(f"{len(m):,} token mappings -> {args.out}")


if __name__ == "__main__":
    main()
