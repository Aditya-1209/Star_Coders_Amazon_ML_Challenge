"""Compare r2 and r5 feature kernels on identical real candidate pairs."""
import argparse
import json
import statistics
import subprocess
import time
import types
from pathlib import Path
import polars as pl
from polars.testing import assert_frame_equal
from er_v2.features import compute, record_frames, pair_frame, add_block_context
from er_v2.run_block import load_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--pairs', type=int, default=100000)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--repeats', type=int, default=3)
    args = ap.parse_args()
    path = 'code/business_entity_resolution/src/er_v2/features.py'
    # Pin the reviewed r2 source. Only its worker count is adjusted for parity.
    original = subprocess.check_output(['git', 'show', f'427aa32:{path}'], text=True, encoding='utf-8')
    module = types.ModuleType('r2_features')
    exec(compile(original.replace('workers=-1', f'workers={args.workers}'), path, 'exec'), module.__dict__)
    s, t = load_split(args.work, 'train')
    left, right = record_frames(s, t)
    c = add_block_context(pl.read_parquet(args.work/'cands_train.parquet')).head(args.pairs)
    n2 = pl.scan_parquet(args.work/'norm/train_source2.parquet').select(pl.len()).collect().item()
    pairs = pair_frame(c, left, right, n2)
    timings = {'r2': [], 'r5': []}
    outputs = {}
    for iteration in range(args.repeats + 1):
        order = ('r2', 'r5') if iteration % 2 else ('r5', 'r2')
        for name in order:
            start = time.perf_counter()
            outputs[name] = module.compute(pairs) if name == 'r2' else compute(pairs, workers=args.workers)
            if iteration:
                timings[name].append(time.perf_counter()-start)
    assert_frame_equal(outputs['r2'], outputs['r5'].select(outputs['r2'].columns), check_exact=True)
    result = {'r2_commit': '427aa32', 'pairs': len(pairs), 'workers': args.workers,
              'seconds': timings, 'median_seconds': {k: statistics.median(v) for k,v in timings.items()},
              'all_original_features_exactly_equal': True,
              'scope': 'Feature kernel only; excludes per-record preparation, blocking, training, and inference.'}
    result['kernel_speedup'] = result['median_seconds']['r2']/result['median_seconds']['r5']
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
