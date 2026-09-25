"""Stream a reproducible development corpus; never use its score as a leaderboard estimate."""
import argparse
import csv
import json
import random
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--anchors', type=int, default=6000)
    ap.add_argument('--distractor-rate', type=float, default=.002)
    args = ap.parse_args()
    out = args.out / 'train'
    out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(20260925)
    t = time.perf_counter()
    def rows(name):
        with (args.dataset / 'train' / name).open(encoding='utf-8', newline='') as fh:
            yield from csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE)
    def write(name, data, fields):
        with (out / name).open('w', encoding='utf-8', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=fields, delimiter='\t', quoting=csv.QUOTE_NONE, lineterminator='\n')
            w.writeheader()
            w.writerows(data)
    sample = []
    for i, row in enumerate(rows('train_source1.tsv')):
        if i < args.anchors:
            sample.append(row)
        else:
            j = rng.randrange(i + 1)
            if j < args.anchors:
                sample[j] = row
    sample.sort(key=lambda r: r['entity_id'])
    fields = list(sample[0])
    write('train_source1.tsv', sample, fields)
    ids = {r['entity_id'] for r in sample}
    gt = [r for r in rows('train_ground_truth.tsv') if r['source1_entity_id'] in ids]
    targets = {x for r in gt for x in r['matched_entity_ids'].split(',') if x}
    write('train_ground_truth.tsv', gt, list(gt[0]))
    counts = {}
    for src in (2, 3):
        name = f'train_source{src}.tsv'
        selected = [r for r in rows(name) if r['entity_id'] in targets or rng.random() < args.distractor_rate]
        write(name, selected, fields)
        counts[name] = len(selected)
        print(name, len(selected), 'elapsed', round(time.perf_counter()-t), flush=True)
    meta = {'seed': 20260925, 'anchors': len(sample), 'true_pairs': len(targets),
            'distractor_rate': args.distractor_rate, 'targets': counts,
            'limitation': 'Reduced distractor corpus; scores and speed are not full-corpus or leaderboard estimates.'}
    (args.out / 'sample.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta), flush=True)


if __name__ == '__main__':
    main()
