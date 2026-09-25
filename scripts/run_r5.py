"""Reproducible r5 desktop runner. Run from any directory with Python 3.12.

Checkpoint reuse requires the same command, source code, and input-file
fingerprints. Interrupted feature generation is deliberately not auto-resumed.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--work', type=Path, default=Path('work/r5'))
    ap.add_argument('--output', type=Path, default=Path('output/r5'))
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--threads', type=positive, default=12)
    ap.add_argument('--shard-pairs', type=positive, default=1_000_000)
    ap.add_argument('--block-chunk', type=positive, default=10_000)
    ap.add_argument('--rescue-k', type=int, default=12)
    ap.add_argument('--rounds', type=positive, default=1500)
    ap.add_argument('--phase', choices=['all', 'train', 'predict'], default='all')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    if args.rescue_k < 0:
        ap.error('--rescue-k must be nonnegative')
    # Resolve caller-supplied paths before changing subprocess cwd.
    dataset, work, output = args.dataset.resolve(), args.work.resolve(), args.output.resolve()
    models = work / 'models'
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT/'code/business_entity_resolution/src') + os.pathsep + env.get('PYTHONPATH', '')
    env['PYTHONUTF8'] = '1'
    env['POLARS_MAX_THREADS'] = str(args.threads)
    env['OMP_NUM_THREADS'] = str(args.threads)
    env['OPENBLAS_NUM_THREADS'] = str(args.threads)
    code = hashlib.sha256()
    for path in sorted((ROOT/'code/business_entity_resolution/src/er_v2').glob('*.py')):
        code.update(path.name.encode())
        code.update(path.read_bytes())
    checkpoints = work / 'checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=True)
    timings = {}

    def fingerprint(paths):
        return [{'path': str(p), 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                for p in paths]

    def stage(name, module_args, inputs, outputs, script=False):
        command = [sys.executable] + ([] if script else ['-m']) + list(map(str, module_args))
        stamp = checkpoints / (name+'.json')
        expected = {'command': command, 'code_sha256': code.hexdigest(), 'inputs': fingerprint(inputs),
                    'threads': args.threads}
        if stamp.exists():
            previous = json.loads(stamp.read_text(encoding='utf-8'))
            matches = previous.get('signature') == expected and all(p.exists() for p in outputs)
            if matches:
                matches = previous.get('outputs') == fingerprint(outputs)
            if args.resume and matches:
                print(f'{name}: reuse completed stage', flush=True)
                return
            raise RuntimeError(f'{name}: existing checkpoint differs or --resume was omitted. Use a fresh --work directory.')
        print(f'\n{name}: starting', flush=True)
        t = time.perf_counter()
        with (checkpoints/(name+'.log')).open('w', encoding='utf-8') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, encoding='utf-8')
            for line in child.stdout:
                print(line, end='', flush=True)
                log.write(line)
            status = child.wait()
        if status:
            raise SystemExit(f'{name} failed (exit {status}); see {checkpoints/(name+".log")}')
        elapsed = time.perf_counter()-t
        record = {'signature': expected, 'outputs': fingerprint(outputs), 'seconds': elapsed}
        stamp.write_text(json.dumps(record, indent=2), encoding='utf-8')
        timings[name] = elapsed

    train_files = [dataset/'train'/f'train_source{i}.tsv' for i in (1, 2, 3)] + [dataset/'train/train_ground_truth.tsv']
    if args.phase in ('all', 'train'):
        stage('translit', ['er_v2.translit', '--dataset', dataset, '--out', models/'translit.json'],
              train_files, [models/'translit.json'])

    for split in ('train', 'test'):
        if (split == 'train' and args.phase == 'predict') or (split == 'test' and args.phase == 'train'):
            continue
        raw = [dataset/split/f'{split}_source{i}.tsv' for i in (1, 2, 3)]
        norm = [work/'norm'/f'{split}_source{i}.parquet' for i in (1, 2, 3)]
        stage(f'normalize-{split}', ['er_v2.prepare', '--dataset', dataset, '--work', work,
              '--workers', args.threads, '--splits', split, '--translit', models/'translit.json'],
              raw+[models/'translit.json'], norm)
        cands = work/f'cands_{split}.parquet'
        stage(f'block-{split}', ['er_v2.run_block', '--work', work, '--split', split,
              '--chunk', args.block_chunk, '--rescue-k', args.rescue_k], norm, [cands])
        extra = [] if split == 'train' else ['--stage1-model-dir', models, '--device', args.device]
        features = work/f'feats_{split}'
        inputs = norm+[cands]+([train_files[-1]] if split == 'train' else [models/'stage1.json', models/'metrics.json'])
        stage(f'features-{split}', ['er_v2.run_features', '--dataset', dataset, '--work', work,
              '--split', split, '--workers', args.threads, '--shard-pairs', args.shard_pairs]+extra,
              inputs, [features/'manifest.json'])
        if split == 'train':
            stage('train', ['er_v2.train', '--dataset', dataset, '--work', work, '--model-dir', models,
                  '--device', args.device, '--threads', args.threads, '--rounds', args.rounds],
                  norm+[features/'manifest.json', train_files[-1], cands],
                  [models/'stage1.json', models/'stage2.json', models/'metrics.json', work/'eval_preds.parquet'])
        else:
            outputs = [output/'matching_results.tsv', output/'candidate_pairs.tsv']
            stage('predict', ['er_v2.predict', '--work', work, '--model-dir', models, '--output', output,
                  '--device', args.device, '--threads', args.threads],
                  norm+[features/'manifest.json', models/'stage1.json', models/'stage2.json', models/'metrics.json'], outputs)
            validator = dataset.parent/'utils/validate_submission.py'
            if not validator.exists():
                raise FileNotFoundError(f'Official validator required: {validator}')
            # Always validate; do not reuse a previous validation result.
            subprocess.run([sys.executable, str(validator), '--matching', str(outputs[0]), '--candidate',
                            str(outputs[1]), '--test-dir', str(dataset/'test'), '--check-ids'], cwd=ROOT, env=env, check=True)
    (work/'last_run_timings.json').write_text(json.dumps(timings, indent=2), encoding='utf-8')
    print(f'Finished. Metrics: {models / "metrics.json"}', flush=True)
    if args.phase != 'train':
        print(f'Submission files: {output}', flush=True)


if __name__ == '__main__':
    main()
