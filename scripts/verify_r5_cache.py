"""End-to-end parity of cached vs ordinary inference using development records.

This is an engineering check, not a second accuracy evaluation.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
import polars as pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--threads', default=2, type=int)
    args = ap.parse_args()
    results = {}
    for mode in ('ordinary', 'cached'):
        work = args.out / mode
        norm = work / 'norm'
        norm.mkdir(parents=True, exist_ok=False)
        for src in (1, 2, 3):
            shutil.copy2(args.work / 'norm' / f'train_source{src}.parquet', norm / f'test_source{src}.parquet')
        shutil.copy2(args.work / 'cands_train.parquet', work / 'cands_test.parquet')
        model = args.work / 'models'
        t = time.perf_counter()
        cmd = [sys.executable, '-m', 'er_v2.run_features', '--work', str(work), '--split', 'test',
               '--shard-pairs', '100000', '--workers', str(args.threads), '--device', 'cpu']
        if mode == 'cached':
            cmd += ['--stage1-model-dir', str(model)]
        subprocess.run(cmd, check=True)
        subprocess.run([sys.executable, '-m', 'er_v2.predict', '--work', str(work), '--model-dir', str(model),
                        '--output', str(work/'output'), '--device', 'cpu', '--threads', str(args.threads),
                        '--batch-rows', '50000'], check=True)
        results[mode] = {'seconds': time.perf_counter()-t,
                         'feature_bytes': sum(p.stat().st_size for p in (work/'feats_test').glob('part_*.parquet')),
                         'cache_bytes': sum(p.stat().st_size for p in (work/'feats_test').glob('scores_*.parquet'))}
    a = pl.read_parquet(args.out/'ordinary/test_preds.parquet').sort('sidx', 'tidx')
    b = pl.read_parquet(args.out/'cached/test_preds.parquet').sort('sidx', 'tidx')
    from polars.testing import assert_frame_equal
    assert_frame_equal(a, b, check_exact=True)
    for name in ('matching_results.tsv', 'candidate_pairs.tsv'):
        assert (args.out/'ordinary/output'/name).read_bytes() == (args.out/'cached/output'/name).read_bytes(), name
    results['predictions_exactly_equal'] = True
    results['candidate_pairs'] = len(a)
    (args.out/'comparison.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
