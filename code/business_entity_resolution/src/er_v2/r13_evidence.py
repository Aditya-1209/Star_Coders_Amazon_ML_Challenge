"""Bounded, label-free neural evidence from other records of the same business.

Only stage-2 predictions select anchors. Every candidate excludes itself from
support; a high score cannot certify its own match. No labels or CE scores enter
this module. Target arrays remain memory mapped, with fixed-size pair gathers.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from .graph import hop_keep
from .r10_retrieval import side_rows, save_json

KEYS = ['sidx', 'tidx']
FEATURES = ['sib_count', 'sib_cos_max', 'sib_cos_mean', 'sib_cos_min',
            'sib_confidence', 'sib_cos_weighted', 'sib_cos_gain']
# These graph/CE target ranks see only folds 3/4/6/7 during training but all
# businesses during inference. R12's global CE join fixes per-stage filtering,
# not this population mismatch. Keep pair-local and within-business evidence.
INCOMPLETE_TARGET_CONTEXT = {'ce_rank_t', 'ce_gap_t', 'ncos_rank_t', 'ncos_gap_t'}


def trusted_anchors(scores, k=3, threshold=.98, margin=.10):
    if k < 1 or not 0 <= threshold <= 1 or not 0 <= margin <= 1:
        raise ValueError('Invalid sibling anchor limits')
    scores = scores.select(*KEYS, 'p2')
    if scores.select(KEYS).n_unique() != len(scores):
        raise ValueError('Duplicate stage-2 pair keys')
    if scores['p2'].null_count() or not scores['p2'].is_finite().all():
        raise ValueError('Non-finite stage-2 anchor scores')
    # Two claimants suffice to establish ownership; ties have zero margin.
    top = scores.sort(['tidx', 'p2', 'sidx'], descending=[False, True, False]).group_by(
        'tidx', maintain_order=True).head(2)
    best = top.group_by('tidx', maintain_order=True).agg(
        pl.col('sidx').first(), pl.col('p2').first(),
        other=pl.col('p2').slice(1).first().fill_null(0.))
    return (best.filter((pl.col('p2') >= threshold) & (pl.col('p2') - pl.col('other') >= margin))
        .sort(['sidx', 'p2', 'tidx'], descending=[False, True, False])
        .group_by('sidx', maintain_order=True).head(k).select(*KEYS, 'p2'))


class SiblingIndex:
    def __init__(self, anchors, embeddings, source_rows, k=3):
        self.emb = embeddings
        self.ids = np.full((source_rows, k), -1, dtype=np.int64)
        self.conf = np.zeros((source_rows, k), dtype=np.float32)
        if anchors.is_empty():
            return
        ranked = anchors.sort(['sidx', 'p2', 'tidx'], descending=[False, True, False]).with_columns(
            slot=pl.col('tidx').cum_count().over('sidx') - 1)
        if ranked['slot'].max() >= k:
            raise ValueError('Too many anchors per business')
        s, t, slot = (ranked[x].to_numpy() for x in ('sidx', 'tidx', 'slot'))
        self.ids[s, slot] = t
        self.conf[s, slot] = ranked['p2'].to_numpy()

    def features(self, pairs, batch_rows=8192):
        if batch_rows < 1:
            raise ValueError('batch_rows must be positive')
        values = np.zeros((len(pairs), len(FEATURES)), dtype=np.float32)
        for start in range(0, len(pairs), batch_rows):
            part = pairs.slice(start, batch_rows)
            s, t = (part[x].to_numpy() for x in KEYS)
            ids, confidence = self.ids[s], self.conf[s]
            valid = (ids >= 0) & (ids != t[:, None])
            # Only gather valid support. Never read a missing/self anchor as evidence.
            count = valid.sum(1)
            cos = np.zeros(ids.shape, dtype=np.float32)
            target = np.asarray(self.emb[t], dtype=np.float32)
            for slot in range(ids.shape[1]):
                take = valid[:, slot]
                if take.any():
                    other = np.asarray(self.emb[ids[take, slot]], dtype=np.float32)
                    cos[take, slot] = np.einsum('ij,ij->i', target[take], other)
            has = count > 0
            maximum = np.where(has, np.where(valid, cos, -np.inf).max(1), 0.)
            minimum = np.where(has, np.where(valid, cos, np.inf).min(1), 0.)
            weight = confidence * valid
            weighted = (cos * weight).sum(1) / np.maximum(weight.sum(1), 1e-12)
            source_cos = part['ncos'].to_numpy()
            values[start:start + len(part)] = np.column_stack([
                count, maximum, cos.sum(1) / np.maximum(count, 1), minimum,
                weight.max(1), weighted, np.where(has, maximum - source_cos, 0.)])
        if not np.isfinite(values).all():
            raise ValueError('Non-finite sibling features/embeddings')
        return pairs.select(KEYS).with_columns([pl.Series(c, values[:, i]) for i, c in enumerate(FEATURES)])


def build(args):
    filename = 'stage2_train.parquet' if args.split == 'train' else 'test_preds.parquet'
    scores = pl.read_parquet(args.work / filename, columns=[*KEYS, 'p2'])
    anchors = trusted_anchors(scores, args.anchors, args.threshold, args.margin)
    del scores
    emb = np.load(args.work / 'emb' / f'{args.split}_tg.npy', mmap_mode='r')
    index = SiblingIndex(anchors, emb, side_rows(args.work, args.split, 's1'), args.anchors)
    path = args.work / f'r13_evidence_{args.split}.parquet'
    temporary = path.with_suffix('.partial.parquet')
    writer = None
    rows = supported = 0
    try:
        source = pq.ParquetFile(args.work / f'stage3_{args.split}.parquet')
        for batch in source.iter_batches(batch_size=args.batch_rows,
                columns=[*KEYS, 'ncos', 'direct', 'sup_both_max', 'sup_addr_valid', 'sup_name_tset']):
            pairs = pl.from_arrow(batch).filter(hop_keep())
            if pairs.is_empty():
                continue
            result = index.features(pairs)
            if writer is None:
                writer = pq.ParquetWriter(temporary, result.to_arrow().schema, compression='zstd')
            writer.write_table(result.to_arrow())
            rows += len(result)
            supported += int((result['sib_count'] > 0).sum())
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        pl.DataFrame(schema={**dict.fromkeys(KEYS, pl.UInt32), **dict.fromkeys(FEATURES, pl.Float32)}).write_parquet(temporary)
    temporary.replace(path)
    save_json(args.work / f'r13_evidence_{args.split}.json', {
        'pairs': rows, 'pairs_with_other_anchor': supported, 'anchors': len(anchors),
        'businesses_with_anchor': anchors['sidx'].n_unique(), 'max_anchors': args.anchors,
        'threshold': args.threshold, 'ownership_margin': args.margin,
        'anchor_source': filename, 'uses_labels': False, 'self_support': False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--split', choices=['train', 'test'], required=True)
    p.add_argument('--anchors', type=int, default=3)
    p.add_argument('--threshold', type=float, default=.98)
    p.add_argument('--margin', type=float, default=.10)
    p.add_argument('--batch-rows', type=int, default=100000)
    args = p.parse_args()
    if args.anchors < 1 or args.batch_rows < 1 or not 0 <= args.threshold <= 1 or not 0 <= args.margin <= 1:
        p.error('Invalid sibling evidence settings')
    build(args)


if __name__ == '__main__':
    main()
