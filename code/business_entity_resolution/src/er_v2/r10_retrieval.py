"""Bounded-memory embedding and country-scoped ANN retrieval for R10."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from .neural import record_text
from .run_block import load_split, split_countries, parquet_rows


def save_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def iter_records(work, split, side, rows=20000):
    """Yield input order, including the global Source 3 offset."""
    offset = 0
    for source in ([1] if side == 's1' else [2, 3]):
        path = work / 'norm' / f'{split}_source{source}.parquet'
        for batch in pq.ParquetFile(path).iter_batches(batch_size=rows, columns=['idx', 'business_name', 'business_address']):
            yield pl.from_arrow(batch).with_columns((pl.col('idx') + offset).cast(pl.UInt32))
        offset += parquet_rows(path)


def side_rows(work, split, side):
    return sum(parquet_rows(work / 'norm' / f'{split}_source{i}.parquet') for i in ([1] if side == 's1' else [2, 3]))


def encode(args):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(args.model_dir), device=args.device)
    model.max_seq_length = args.max_length
    if args.device == 'cuda':
        model.half()
    out = args.work / 'emb'
    out.mkdir(exist_ok=True)
    for side in ('s1', 'tg'):
        path = out / f'{args.split}_{side}.npy'
        temporary = path.with_suffix('.partial.npy')
        arr = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float16,
            shape=(side_rows(args.work, args.split, side), model.get_sentence_embedding_dimension()))
        done = 0
        for frame in iter_records(args.work, args.split, side):
            # Length-sort only one window; no 10M-element Python string list.
            texts = record_text(frame)
            order = np.argsort([len(s) for s in texts], kind='stable')
            values = model.encode([texts[i] for i in order], batch_size=args.batch,
                                 normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            arr[frame['idx'].to_numpy()[order]] = values.astype(np.float16)
            done += len(frame)
            if done % 200000 == 0:
                print(f'encode {args.split}/{side}: {done:,}', flush=True)
        arr.flush()
        del arr
        temporary.replace(path)
    save_json(out / f'{args.split}.json', {'model': str(args.model_dir), 'max_length': args.max_length,
        'dtype': 'float16', 'normalized': True})


def unit(values):
    x = np.ascontiguousarray(values, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def make_index(embeddings, ids, nlist, nprobe, sample_size, threads):
    """SQ8 stores d+8 bytes/vector. Tiny sets use an exact index for testing."""
    import faiss
    faiss.omp_set_num_threads(threads)
    dim = embeddings.shape[1]
    if len(ids) < 10000:
        index = faiss.IndexFlatIP(dim)
    else:
        cells = min(nlist, max(1, len(ids) // 80), max(1, sample_size // 40))
        index = faiss.IndexIVFScalarQuantizer(faiss.IndexFlatIP(dim), dim, cells,
            faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
        index.cp.seed = 1010
        sample = np.random.default_rng(1010).choice(ids, min(sample_size, len(ids)), replace=False)
        index.train(unit(embeddings[sample]))
        index.nprobe = min(nprobe, cells)
    for i in range(0, len(ids), 20000):
        index.add(unit(embeddings[ids[i:i + 20000]]))
    return index


def rerank(queries, targets, ids, approximate, k):
    """Exact cosine on the ANN shortlist; -1 means an unfilled ANN slot."""
    valid = approximate >= 0
    positions = np.maximum(approximate, 0)
    vectors = unit(targets[ids[positions]])
    scores = np.einsum('bd,bkd->bk', unit(queries), vectors)
    scores[~valid] = -np.inf
    # Stable ordering makes ties reproducible, including padded/duplicate texts.
    order = np.argsort(-scores, axis=1, kind='stable')[:, :k]
    return np.take_along_axis(scores, order, axis=1), ids[np.take_along_axis(positions, order, axis=1)]


def exact_topk(queries, targets, ids, k, chunk=20000):
    """Small-query audit only; never form the all-businesses x all-records matrix."""
    q = unit(queries)
    best = np.full((len(q), k), -np.inf, np.float32)
    selected = np.full((len(q), k), -1, np.int64)
    for i in range(0, len(ids), chunk):
        ti = ids[i:i + chunk]
        score = q @ unit(targets[ti]).T
        pool = np.concatenate([best, score], axis=1)
        candidates = np.concatenate([selected, np.broadcast_to(ti, score.shape)], axis=1)
        order = np.argsort(-pool, axis=1, kind='stable')[:, :k]
        best = np.take_along_axis(pool, order, axis=1)
        selected = np.take_along_axis(candidates, order, axis=1)
    return selected


def block(args):
    es = np.load(args.work / 'emb' / f'{args.split}_s1.npy', mmap_mode='r')
    et = np.load(args.work / 'emb' / f'{args.split}_tg.npy', mmap_mode='r')
    path = args.work / f'ncands_{args.split}.parquet'
    temporary = path.with_suffix('.partial.parquet')
    writer = None
    audit = {}
    try:
        for country in split_countries(args.work, args.split):
            s1, tg = load_split(args.work, args.split, country, ['idx', 'country'])
            si, ti = s1['idx'].to_numpy(), tg['idx'].to_numpy()
            if not len(ti):
                continue
            index = make_index(et, ti, args.nlist, args.nprobe, args.index_sample, args.threads)
            k = min(args.k, len(ti))
            search_k = min(max(args.search_k, k), len(ti))
            if args.audit_queries:
                qs = np.random.default_rng(10).choice(si, min(args.audit_queries, len(si)), replace=False)
                _, found = index.search(unit(es[qs]), search_k)
                _, chosen = rerank(es[qs], et, ti, found, k)
                exact = exact_topk(es[qs], et, ti, k)
                recall = float(np.mean([len(set(a) & set(b)) / k for a, b in zip(chosen, exact)]))
                audit[country] = {'queries': len(qs), 'ann_recall_at_k': recall, 'k': k}
                print(f'{country}: ANN vs exact recall@{k}={recall:.4f}', flush=True)
                if recall < args.minimum_ann_recall:
                    raise ValueError(f'{country} ANN recall {recall:.3f} below gate; increase --nprobe/--search-k in a fresh run')
            for i in range(0, len(si), args.query_batch):
                ids = si[i:i + args.query_batch]
                _, found = index.search(unit(es[ids]), search_k)
                scores, targets = rerank(es[ids], et, ti, found, k)
                frame = pl.DataFrame({'sidx': np.repeat(ids, k).astype(np.uint32),
                    'tidx': targets.ravel().astype(np.uint32), 'ncos': scores.ravel().astype(np.float32),
                    'nrank': np.tile(np.arange(1, k + 1, dtype=np.uint16), len(ids))}).filter(pl.col('ncos').is_finite())
                if writer is None:
                    writer = pq.ParquetWriter(temporary, frame.to_arrow().schema, compression='zstd')
                writer.write_table(frame.to_arrow())
            print(f'ANN {args.split}/{country}: {len(si):,} queries, {len(ti):,} targets', flush=True)
            del index, s1, tg
            gc.collect()
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        pl.DataFrame(schema={'sidx': pl.UInt32, 'tidx': pl.UInt32, 'ncos': pl.Float32, 'nrank': pl.UInt16}).write_parquet(temporary)
    temporary.replace(path)
    save_json(args.work / f'ann_{args.split}.json', {'countries': audit, 'nprobe': args.nprobe,
        'nlist': args.nlist, 'search_k': args.search_k, 'k': args.k, 'approximate': True})


def merge(args):
    """Keep lexical input immutable so stage resume has stable fingerprints."""
    base = pl.scan_parquet(args.work / f'key_{args.split}.parquet')
    schema = base.collect_schema()
    neural = pl.scan_parquet(args.work / f'ncands_{args.split}.parquet').filter(pl.col('nrank') <= args.k)
    new = neural.select('sidx', 'tidx').join(base.select('sidx', 'tidx'), on=['sidx', 'tidx'], how='anti')
    new = new.with_columns([pl.lit(999 if c == 'brank' else 0).cast(t).alias(c)
                            for c, t in schema.items() if c not in ('sidx', 'tidx')]).select(schema.names())
    out = args.work / f'cands_{args.split}.parquet'
    tmp = out.with_suffix('.partial.parquet')
    pl.concat([base, new]).sort('sidx', 'brank', 'tidx').sink_parquet(tmp)
    tmp.replace(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['encode', 'block', 'merge'])
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--model-dir', type=Path)
    p.add_argument('--split', choices=['train', 'test'], required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--max-length', type=int, default=64)
    p.add_argument('--k', type=int, default=24)
    p.add_argument('--search-k', type=int, default=96)
    p.add_argument('--nlist', type=int, default=4096)
    p.add_argument('--nprobe', type=int, default=96)
    p.add_argument('--index-sample', type=int, default=200000)
    p.add_argument('--query-batch', type=int, default=128)
    p.add_argument('--threads', type=int, default=12)
    p.add_argument('--audit-queries', type=int, default=32)
    p.add_argument('--minimum-ann-recall', type=float, default=0.95)
    args = p.parse_args()
    if min(args.batch, args.max_length, args.k, args.search_k, args.nlist, args.nprobe, args.index_sample, args.query_batch, args.threads) < 1 or args.k >= 65535:
        p.error('Counts must be positive; k must fit uint16')
    if args.audit_queries < 0 or not 0 <= args.minimum_ann_recall <= 1:
        p.error('Invalid ANN audit settings')
    globals()[args.stage](args)

if __name__ == '__main__':
    main()
