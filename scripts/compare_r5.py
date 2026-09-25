"""Prepare a controlled r2-feature reference from a fresh r5 development run."""
import argparse
import json
import shutil
from pathlib import Path
import polars as pl
from er_v2.features import add_block_context
from er_v2.metrics import macro_f05
from er_v2.run_block import load_split
from er_v2.run_features import ground_truth_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--dataset', type=Path, required=True)
    args = ap.parse_args()
    args.reference.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.work / 'norm', args.reference / 'norm')
    c = pl.read_parquet(args.work / 'cands_train.parquet')
    base = c.filter(pl.col('rescue') == 0).drop('rescue')
    base.write_parquet(args.reference / 'cands_train.parquet')
    ctx = add_block_context(base)
    context_cols = [x for x in ctx.columns if x not in base.columns]
    folder = args.reference / 'feats_train'
    folder.mkdir()
    for path in sorted((args.work / 'feats_train').glob('part_*.parquet')):
        data = pl.read_parquet(path).filter(pl.col('rescue') == 0).drop('rescue', *context_cols)
        data = data.join(ctx.select('sidx', 'tidx', *context_cols), on=['sidx', 'tidx'], how='left')
        # Match original r2 feature order, not the join's new column order.
        original = pl.read_parquet(path, n_rows=0).columns
        data.select([x for x in original if x != 'rescue']).write_parquet(folder / path.name)
    s, t = load_split(args.work, 'train')
    gt = ground_truth_pairs(args.dataset, s, t)
    report = {'anchors': len(s), 'targets': len(t), 'true_pairs': len(gt),
              'base_pairs': len(base), 'r5_pairs': len(c),
              'base_oracle': macro_f05(base.join(gt, on=['sidx', 'tidx']), gt, s['idx']),
              'r5_oracle': macro_f05(c.join(gt, on=['sidx', 'tidx']), gt, s['idx'])}
    (args.work / 'retrieval_comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
