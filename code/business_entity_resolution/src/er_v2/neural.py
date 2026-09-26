"""Neural matching signal: a fine-tuned multilingual sentence encoder.

Model: intfloat/multilingual-e5-small (MIT licence, 118M parameters). It reads
Latin, Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati, Malayalam, Oriya
and Gurmukhi natively, so transliterated Indian names need no hand-made mapping.

Leakage rule: the encoder is fine-tuned only on businesses from the stage-1
folds (0, 1, 8, 9). Stage 2 (folds 2/5), stage 3 (folds 6/7) and the tuning and
holdout folds (3/4) therefore see neural scores the encoder never trained on.

Commands (run from the repository root, PYTHONPATH=code/business_entity_resolution/src):
  python -m er_v2.neural finetune --pairs 1000000
  python -m er_v2.neural encode --split train      (and --split test)
  python -m er_v2.neural block --split train --k 32 (and --split test)
  python -m er_v2.neural probe                      (retrieval recall on fold 3)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl

BASE_MODEL = "intfloat/multilingual-e5-small"
ENCODER_FOLDS = [0, 1, 8, 9]
MAX_LEN = 48
DIM = 384


def record_text(df: pl.DataFrame) -> list[str]:
    """Raw name and address in their original scripts; e5 expects a task prefix."""
    return [f"query: {n} | {a}" for n, a in zip(df["business_name"].fill_null("").to_list(),
                                                 df["business_address"].fill_null("").to_list())]


def _log(t0: float, msg: str) -> None:
    print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)


# ------------------------------------------------------------------ fine-tune
def finetune(work: Path, dataset: Path, out: Path, n_pairs: int, batch: int, epochs: int, lr: float) -> None:
    """Contrastive fine-tuning: a business and its records close, look-alikes apart.

    Loss: multiple-negatives ranking (in-batch negatives) plus one hard negative
    per pair: the highest-ranked blocking candidate that is not a true match.
    """
    import torch
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    from .run_block import load_split
    from .run_features import ground_truth_pairs
    from .train import fold_expr

    t0 = time.time()
    s1, tg = load_split(work, "train")
    s1 = s1.with_columns(pl.col("idx").cast(pl.UInt32))
    tg = tg.with_columns(pl.col("idx").cast(pl.UInt32))
    truth = ground_truth_pairs(dataset, s1, tg).with_columns(fold_expr())
    truth = truth.filter(pl.col("fold").is_in(ENCODER_FOLDS)).drop("fold")
    pos = truth.sample(min(n_pairs, len(truth)), seed=7, shuffle=True)
    # hard negative: best-ranked blocking candidate of the same business that is not a match
    cands = (pl.scan_parquet(work / "cands_train.parquet").select("sidx", "tidx", "brank")
             .join(pos.select("sidx").unique().lazy(), on="sidx", how="semi").collect())
    neg = (cands.join(truth, on=["sidx", "tidx"], how="anti").sort("sidx", "brank")
           .unique("sidx", keep="first").select("sidx", neg="tidx"))
    pos = pos.join(neg, on="sidx", how="inner")
    text_s = dict(zip(s1["idx"].to_list(), record_text(s1)))
    need = set(pos["tidx"].to_list()) | set(pos["neg"].to_list())
    tsub = tg.filter(pl.col("idx").is_in(list(need)))
    text_t = dict(zip(tsub["idx"].to_list(), record_text(tsub)))
    examples = [InputExample(texts=[text_s[s], text_t[t], text_t[n]])
                for s, t, n in zip(pos["sidx"].to_list(), pos["tidx"].to_list(), pos["neg"].to_list())]
    _log(t0, f"{len(examples):,} (business, match, hard negative) triples from folds {ENCODER_FOLDS}")

    model = SentenceTransformer(BASE_MODEL, device="cuda")
    model.max_seq_length = MAX_LEN
    loader = DataLoader(examples, shuffle=True, batch_size=batch, drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(model)
    steps = len(loader) * epochs
    model.fit(train_objectives=[(loader, loss)], epochs=epochs, warmup_steps=int(0.05 * steps),
              optimizer_params={"lr": lr}, use_amp=True, show_progress_bar=True)
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    (out / "finetune.json").write_text(json.dumps({
        "base_model": BASE_MODEL, "licence": "MIT", "pairs": len(examples), "batch": batch,
        "epochs": epochs, "lr": lr, "encoder_folds": ENCODER_FOLDS, "max_len": MAX_LEN}, indent=2))
    _log(t0, f"saved fine-tuned encoder to {out}")


# ------------------------------------------------------------------ encode
def encode(work: Path, model_dir: Path, split: str, batch: int) -> None:
    """Embed every Source 1 and Source 2/3 record of a split (float16, L2-normalised)."""
    import torch
    from sentence_transformers import SentenceTransformer

    from .run_block import load_split

    t0 = time.time()
    model = SentenceTransformer(str(model_dir), device="cuda")
    model.max_seq_length = MAX_LEN
    model.half()
    s1, tg = load_split(work, split)
    out = work / "emb"
    out.mkdir(exist_ok=True)
    for side, df in (("s1", s1), ("tg", tg)):
        texts = record_text(df)
        order = np.argsort([len(x) for x in texts])  # length-sorted batches: far less padding
        emb = np.lib.format.open_memmap(out / f"{split}_{side}.npy", mode="w+", dtype=np.float16,
                                        shape=(len(texts), DIM))
        step = batch * 20
        for i in range(0, len(order), step):
            idx = order[i:i + step]
            with torch.inference_mode():
                v = model.encode([texts[j] for j in idx], batch_size=batch, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False)
            emb[idx] = v.astype(np.float16)
            if (i // step) % 50 == 0:
                _log(t0, f"{split}/{side}: {min(i + step, len(order)):,}/{len(order):,}")
        emb.flush()
        del emb
        _log(t0, f"{split}/{side}: done ({len(texts):,} records)")


# ------------------------------------------------------------------ neural blocking
def block(work: Path, split: str, k: int, qbatch: int) -> None:
    """Top-k nearest Source 2/3 records per business by cosine, within each country (GPU)."""
    import torch

    from .run_block import load_split

    t0 = time.time()
    s1, tg = load_split(work, split)
    es = np.load(work / "emb" / f"{split}_s1.npy", mmap_mode="r")
    et = np.load(work / "emb" / f"{split}_tg.npy", mmap_mode="r")
    parts = []
    for country in s1["country"].unique().sort():
        si = s1.filter(pl.col("country") == country)["idx"].to_numpy()
        ti = tg.filter(pl.col("country") == country)["idx"].to_numpy()
        if len(ti) == 0:
            continue
        T = torch.from_numpy(np.ascontiguousarray(et[ti])).cuda()
        kk = min(k, len(ti))
        # the (queries x records) score matrix must fit in ~2.5 GB of VRAM (fp16)
        qb = max(64, min(qbatch, int(2.5e9 / (2 * len(ti)))))
        for i in range(0, len(si), qb):
            q = torch.from_numpy(np.ascontiguousarray(es[si[i:i + qb]])).cuda()
            val, pos = torch.topk(q @ T.T, kk, dim=1)
            val, pos = val.float().cpu().numpy(), pos.cpu().numpy()
            parts.append(pl.DataFrame({
                "sidx": np.repeat(si[i:i + qb], kk).astype(np.uint32),
                "tidx": ti[pos.ravel()].astype(np.uint32),
                "ncos": val.ravel().astype(np.float32),
                "nrank": np.tile(np.arange(1, kk + 1, dtype=np.uint16), len(q)),
            }))
        _log(t0, f"{split}/{country}: {len(si):,} businesses x {len(ti):,} records")
        del T
        torch.cuda.empty_cache()
    out = pl.concat(parts)
    out.write_parquet(work / f"ncands_{split}.parquet")
    _log(t0, f"{split}: {len(out):,} neural candidate pairs -> ncands_{split}.parquet")


# ------------------------------------------------------------------ pair scores
class Embeddings:
    """Random access to a split's stored embeddings (loaded once into RAM)."""

    def __init__(self, work: Path, split: str):
        self.s1 = np.load(work / "emb" / f"{split}_s1.npy")
        self.tg = np.load(work / "emb" / f"{split}_tg.npy")

    def cos(self, sidx: np.ndarray, tidx: np.ndarray, chunk: int = 1_000_000) -> np.ndarray:
        import torch
        out = np.empty(len(sidx), dtype=np.float32)
        for i in range(0, len(sidx), chunk):
            a = torch.from_numpy(self.s1[sidx[i:i + chunk]]).cuda()
            b = torch.from_numpy(self.tg[tidx[i:i + chunk]]).cuda()
            out[i:i + chunk] = (a * b).sum(1).float().cpu().numpy()
        return out


NEURAL_FEATURES = ["ncos", "ncos_rank_s", "ncos_gap_s", "ncos_rank_t", "ncos_gap_t"]


def neural_features(pairs: pl.DataFrame, emb: Embeddings) -> pl.DataFrame:
    """Raw cosine plus relative versions (within the business / among a record's claimants).

    Relative features transfer across splits better than raw scores. Never used by
    stage 1: the encoder was trained on stage-1 folds.
    """
    c = emb.cos(pairs["sidx"].to_numpy().astype(np.int64), pairs["tidx"].to_numpy().astype(np.int64))
    df = pairs.select("sidx", "tidx").with_columns(ncos=pl.Series(c))
    return df.with_columns(
        ncos_rank_s=pl.col("ncos").rank("ordinal", descending=True).over("sidx").cast(pl.UInt16),
        ncos_gap_s=(pl.col("ncos").max().over("sidx") - pl.col("ncos")).cast(pl.Float32),
        ncos_rank_t=pl.col("ncos").rank("ordinal", descending=True).over("tidx").cast(pl.UInt16),
        ncos_gap_t=(pl.col("ncos").max().over("tidx") - pl.col("ncos")).cast(pl.Float32),
    )


def merge_candidates(work: Path, split: str, k: int) -> None:
    """Union key-blocking candidates with the neural top-k (flag neural=1 for new pairs)."""
    key_only = work / f"cands_{split}_keyonly.parquet"
    if not key_only.exists():  # keep the pristine key-blocking candidates; merging is idempotent
        import shutil
        shutil.copy2(work / f"cands_{split}.parquet", key_only)
    base = pl.read_parquet(key_only)
    neu = pl.read_parquet(work / f"ncands_{split}.parquet").filter(pl.col("nrank") <= k).select("sidx", "tidx")
    new = neu.join(base.select("sidx", "tidx"), on=["sidx", "tidx"], how="anti")
    fill = {c: pl.lit(0).cast(t) for c, t in base.schema.items() if c not in ("sidx", "tidx")}
    new = new.with_columns(**fill).with_columns(brank=pl.lit(999, base.schema["brank"]))
    out = pl.concat([base.with_columns(neural=pl.lit(0, pl.Int8)),
                     new.select(base.columns).with_columns(neural=pl.lit(1, pl.Int8))])
    out.sort("sidx", "brank").write_parquet(work / f"cands_{split}.parquet")
    print(f"{split}: +{len(new):,} neural-only pairs ({len(new) / base['sidx'].n_unique():.2f} per business); "
          f"total {len(out):,}")


# ------------------------------------------------------------------ cross-encoder
CE_DIR = "work/neural_ce"


def pair_text(s1: pl.DataFrame, tg: pl.DataFrame, pairs: pl.DataFrame) -> tuple[list[str], list[str]]:
    a = pairs.join(s1.select(sidx=pl.col("idx").cast(pl.UInt32), n1="business_name", a1="business_address"),
                   on="sidx", how="left", maintain_order="left")
    a = a.join(tg.select(tidx=pl.col("idx").cast(pl.UInt32), n2="business_name", a2="business_address"),
               on="tidx", how="left", maintain_order="left").fill_null("")
    left = [f"{n} | {ad}" for n, ad in zip(a["n1"].to_list(), a["a1"].to_list())]
    right = [f"{n} | {ad}" for n, ad in zip(a["n2"].to_list(), a["a2"].to_list())]
    return left, right


def ce_finetune(work: Path, out: Path, n_pairs: int, batch: int, lr: float) -> None:
    """Cross-encoder on stage-1 survivors of the encoder folds (0/1/8/9) only."""
    from sentence_transformers.cross_encoder import CrossEncoder
    from sentence_transformers import InputExample
    from torch.utils.data import DataLoader

    from .run_block import load_split
    from .train import fold_expr

    t0 = time.time()
    st2 = pl.read_parquet(work / "stage2_train.parquet", columns=["sidx", "tidx", "label"])
    st2 = st2.with_columns(fold_expr()).filter(pl.col("fold").is_in(ENCODER_FOLDS))
    st2 = st2.sample(min(n_pairs, len(st2)), seed=11, shuffle=True)
    s1, tg = load_split(work, "train")
    left, right = pair_text(s1, tg, st2)
    ex = [InputExample(texts=[l, r], label=float(y)) for l, r, y in zip(left, right, st2["label"].to_list())]
    _log(t0, f"{len(ex):,} survivor pairs ({st2['label'].mean():.2%} matches) from folds {ENCODER_FOLDS}")
    model = CrossEncoder(BASE_MODEL, num_labels=1, max_length=2 * MAX_LEN, device="cuda")
    loader = DataLoader(ex, shuffle=True, batch_size=batch, drop_last=True)
    model.fit(train_dataloader=loader, epochs=1, warmup_steps=int(0.05 * len(loader)),
              optimizer_params={"lr": lr}, use_amp=True, show_progress_bar=True)
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    _log(t0, f"saved cross-encoder to {out}")


def ce_score(work: Path, model_dir: Path, split: str, pairs_path: Path, out_path: Path, batch: int) -> None:
    """Score (sidx, tidx) pairs with the cross-encoder -> parquet (sidx, tidx, ce)."""
    import torch
    from sentence_transformers.cross_encoder import CrossEncoder

    from .run_block import load_split

    t0 = time.time()
    pairs = pl.read_parquet(pairs_path, columns=["sidx", "tidx"]).unique()
    s1, tg = load_split(work, split)
    model = CrossEncoder(str(model_dir), device="cuda", max_length=2 * MAX_LEN)
    model.model.half()
    outs = []
    step = 500_000
    for i in range(0, len(pairs), step):
        chunk = pairs.slice(i, step)
        left, right = pair_text(s1, tg, chunk)
        with torch.inference_mode():
            sc = model.predict(list(zip(left, right)), batch_size=batch, show_progress_bar=False,
                               convert_to_numpy=True)
        outs.append(chunk.with_columns(ce=pl.Series(np.asarray(sc, dtype=np.float32))))
        _log(t0, f"{split}: {min(i + step, len(pairs)):,}/{len(pairs):,} pairs")
    pl.concat(outs).write_parquet(out_path)


# ------------------------------------------------------------------ probe
def probe(work: Path, dataset: Path) -> None:
    """Fold-3 retrieval: how many true pairs does the neural channel add to key blocking?"""
    from .run_block import load_split
    from .run_features import ground_truth_pairs
    from .train import fold_expr

    s1, tg = load_split(work, "train")
    truth = ground_truth_pairs(dataset, s1, tg).with_columns(fold_expr()).filter(pl.col("fold") == 3).drop("fold")
    keyb = pl.scan_parquet(work / "cands_train.parquet").select("sidx", "tidx").join(
        truth.select("sidx").unique().lazy(), on="sidx", how="semi").collect()
    neu = pl.read_parquet(work / "ncands_train.parquet").join(truth.select("sidx").unique(), on="sidx", how="semi")
    n = len(truth)
    hit_key = truth.join(keyb, on=["sidx", "tidx"], how="semi").height
    res = {"true_pairs": n, "key_recall": hit_key / n}
    for k in (8, 16, 32):
        nk = neu.filter(pl.col("nrank") <= k).select("sidx", "tidx")
        hit_n = truth.join(nk, on=["sidx", "tidx"], how="semi").height
        both = truth.join(pl.concat([keyb, nk]).unique(), on=["sidx", "tidx"], how="semi").height
        res[f"neural@{k}_recall"] = hit_n / n
        res[f"union@{k}_recall"] = both / n
        res[f"neural@{k}_new_candidates_per_business"] = nk.join(keyb, on=["sidx", "tidx"], how="anti").height / truth["sidx"].n_unique()
    (work / "neural_probe.json").write_text(json.dumps(res, indent=2))
    for key, v in res.items():
        print(f"{key:40s} {v:.4f}" if isinstance(v, float) else f"{key:40s} {v:,}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["finetune", "encode", "block", "probe", "merge", "ce_finetune", "ce_score"])
    ap.add_argument("--pairs-path", default=None, help="ce_score: parquet with sidx, tidx")
    ap.add_argument("--out-path", default=None, help="ce_score: output parquet")
    ap.add_argument("--work", default="work")
    ap.add_argument("--dataset", default="student_resource/dataset")
    ap.add_argument("--model-dir", default="work/neural_e5")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--pairs", type=int, default=1_000_000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--qbatch", type=int, default=4096)
    args = ap.parse_args()
    work = Path(args.work)
    if args.cmd == "finetune":
        finetune(work, Path(args.dataset), Path(args.model_dir), args.pairs, args.batch, args.epochs, args.lr)
    elif args.cmd == "encode":
        encode(work, Path(args.model_dir), args.split, batch=max(args.batch, 512))
    elif args.cmd == "block":
        block(work, args.split, args.k, args.qbatch)
    elif args.cmd == "merge":
        merge_candidates(work, args.split, args.k)
    elif args.cmd == "ce_finetune":
        ce_finetune(work, Path(CE_DIR), args.pairs, args.batch, args.lr)
    elif args.cmd == "ce_score":
        ce_score(work, Path(CE_DIR), args.split, Path(args.pairs_path), Path(args.out_path), max(args.batch, 256))
    else:
        probe(work, Path(args.dataset))


if __name__ == "__main__":
    main()
