"""R10 regression tests and an offline, real neural/tree/ANN end-to-end smoke test."""
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import polars as pl

from er_v2.r10 import attach_ce, blend, gate, per_business, half
from er_v2.r10_ce import samples, TRAIN_FOLDS, VALID_FOLDS
from er_v2.r10_retrieval import make_index, unit, rerank, exact_topk
from er_v2.train import fold_expr, keep_candidates

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_r10


def frame(data):
    return pl.DataFrame(data).with_columns(pl.col('sidx', 'tidx').cast(pl.UInt32))


class R10UnitTests(unittest.TestCase):
    def test_rescue_is_keyed_and_does_not_keep_every_low_score(self):
        pred = frame({'sidx': [0, 0, 1], 'tidx': [0, 1, 2], 'p1': [.9, .00001, .00001]})
        rescue = pred.slice(1, 1).select('sidx', 'tidx')
        kept = keep_candidates(pred, .001, rescue)
        self.assertEqual(kept['tidx'].to_list(), [0, 1])
        self.assertEqual(keep_candidates(pred, .001)['tidx'].to_list(), [0])

    def test_ce_join_rejects_missing_duplicate_and_nonfinite_scores(self):
        f = frame({'sidx': [1, 1, 2], 'tidx': [0, 1, 1]})
        scores = f.with_columns(ce_logit=pl.Series([2., 1., 3.]))
        actual = attach_ce(f, scores.reverse()).sort('sidx', 'tidx')
        self.assertEqual(actual['ce_logit'].to_list(), [2., 1., 3.])
        self.assertEqual(actual['ce_gap_t'].to_list(), [0., 2., 0.])
        for broken in (scores.head(2), pl.concat([scores, scores.head(1)]), scores.with_columns(ce_logit=pl.lit(float('nan')))):
            with self.assertRaises(ValueError):
                attach_ce(f, broken)

    def test_blend_uses_keys_and_rejects_candidate_mismatch(self):
        a = frame({'sidx': [1, 2], 'tidx': [0, 1], 'score': [.2, .8]})
        b = a.reverse().with_columns(score=1 - pl.col('score'))
        np.testing.assert_allclose(blend(a, b, .5)['score'], [.5, .5])
        with self.assertRaises(ValueError):
            blend(a, b.head(1), .5)

    def test_macro_weighting_includes_empty_singletons_and_gate_rejects_ties(self):
        target = frame({'sidx': [0, 0], 'tidx': [0, 1]})
        country = pl.DataFrame({'sidx': [0, 1], 'country': ['US', 'US']}).with_columns(pl.col('sidx').cast(pl.UInt32))
        np.testing.assert_allclose(per_business(target, target, country['sidx']).sort('sidx')['f'], [1, 1])
        self.assertFalse(gate(target, target, target, country, 0.)['passed'])
        wrong = frame({'sidx': [1], 'tidx': [3]})
        self.assertLess(gate(wrong, target, target, country, 0.)['gain'], 0)

    def test_ann_shortlist_rerank_and_unfilled_slots(self):
        rng = np.random.default_rng(10)
        targets = unit(rng.normal(size=(80, 12)))
        ids = np.array([2, 4, 7, 20, 55, 79])
        queries = targets[ids[:3]]
        index = make_index(targets, ids, 4, 4, 100, 1)
        _, positions = index.search(queries, 5)
        scores, found = rerank(queries, targets, ids, positions, 3)
        np.testing.assert_array_equal(found, exact_topk(queries, targets, ids, 3, chunk=2))
        np.testing.assert_array_equal(found[:, 0], ids[:3])
        positions[:, -1] = -1
        scores, _ = rerank(queries, targets, ids, positions, 5)
        self.assertTrue(np.isneginf(scores[:, -1]).all())

    def test_compressed_ivf_path_against_exact_in_separate_process(self):
        script = """
import numpy as np
from er_v2.r10_retrieval import unit, make_index, rerank, exact_topk
x = unit(np.random.default_rng(11).normal(size=(10000, 16)))
ids = np.arange(len(x))
index = make_index(x, ids, 32, 32, 2000, 2)
assert type(index).__name__ == 'IndexIVFScalarQuantizer'
q = x[[7, 150, 9999]]
_, positions = index.search(q, 64)
_, found = rerank(q, x, ids, positions, 4)
np.testing.assert_array_equal(found, exact_topk(q, x, ids, 4))
"""
        result = subprocess.run([sys.executable, '-c', script], env=os.environ,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_runner_defaults_selection_before_holdout_and_immutable_lexical_outputs(self):
        args = run_r10.parser().parse_args([])
        plan = run_r10.commands(args)
        self.assertLess(list(plan).index('select'), list(plan).index('evaluate'))
        self.assertEqual((args.threads, args.ce_batch, args.encode_batch, args.rescue_k), (12, 16, 256, 8))
        self.assertIn('--candidate-prefix', plan['key_train'][0])
        self.assertNotIn(args.work / 'cands_train.parquet', plan['key_train'][1])
        self.assertIn('--neural-rescue-k', plan['train'][0])

    def test_runner_resume_preserves_completed_stage_and_rejects_changed_artifact(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            work, data, out = root / 'work', root / 'data', root / 'out'
            for split in ('train', 'test'):
                (data / split).mkdir(parents=True)
                (data / split / 'input.tsv').write_text('fixture')
            ready = root / 'ready'
            first = work / 'first.txt'
            argv = ['run_r10.py', '--work', str(work), '--dataset', str(data), '--output', str(out), '--reserve-gb', '1']
            create = 'from pathlib import Path; Path(' + repr(str(first)) + ').write_text("first")'
            second = 'from pathlib import Path; import sys; sys.exit(0 if Path(' + repr(str(ready)) + ').exists() else 2)'
            plan = {'first': ([sys.executable, '-c', create], [first]),
                    'second': ([sys.executable, '-c', second], [])}
            with patch.object(run_r10, 'preflight', return_value={}), patch.object(run_r10, 'code_hash', return_value='code'), \
                 patch.object(run_r10, 'commands', return_value=plan), patch.object(sys, 'argv', argv):
                with self.assertRaisesRegex(RuntimeError, 'second failed'):
                    run_r10.main()
                original_time = first.stat().st_mtime_ns
                ready.write_text('ready')
                # Supply only report inputs; completed work must be reused on resume.
                out.mkdir()
                for name in ('matching_results.tsv', 'candidate_pairs.tsv'):
                    (out / name).write_text('fixture')
                (work / 'metrics.json').write_text(json.dumps({'selected': 'reference', 'local_fold4': {'macro_f05': .9}}))
                with patch.object(sys, 'argv', argv + ['--resume']):
                    run_r10.main()
                self.assertEqual(first.stat().st_mtime_ns, original_time)
                self.assertEqual(json.loads((work / 'run.json').read_text())['status'], 'complete')
                first.write_text('changed')
                with patch.object(sys, 'argv', argv + ['--resume']):
                    with self.assertRaisesRegex(RuntimeError, 'Completed stage changed'):
                        run_r10.main()
                self.assertEqual(json.loads((work / 'result.json').read_text())['status'], 'failed')

    def test_fit_never_passes_holdout_or_gate_labels_to_optimizer(self):
        from er_v2 import r10
        with TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / 'models').mkdir()
            ids = pl.DataFrame({'sidx': np.arange(200, dtype=np.uint32)}).with_columns(fold_expr())
            rows = ids.with_columns(tidx=pl.col('sidx'), label=(pl.col('sidx') % 2).cast(pl.Int8), ce_logit=pl.lit(.1))
            seen = []
            class Model:
                best_iteration = 1
                def save_model(self, path):
                    Path(path).write_text('{}')
            def fit(tr, features, va, *more):
                seen.append((tr, va))
                self.assertTrue(set(tr['fold']) <= {6, 7})
                self.assertEqual(set(va['fold']), {3})
                self.assertTrue(va.select(half())[:, 0].eq(0).all())
                self.assertNotIn('label', features)
                return Model()
            with patch.object(r10, 'frame_for', return_value=rows), patch.object(r10, 'fit', side_effect=fit):
                r10.fit_final(SimpleNamespace(work=work, rounds=2, threads=2, device='cpu'))
            self.assertEqual(len(seen), 2)


class R10PipelineTest(unittest.TestCase):
    def test_offline_neural_ann_graph_fusion_and_official_format(self):
        # No model download: a tiny real BERT tests gradients, serialization and token pairing.
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            work, dataset, model, base = root / 'work', root / 'data', root / 'work/models', root / 'tiny'
            (work / 'norm').mkdir(parents=True)
            model.mkdir()
            base.mkdir()
            vocabulary = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'acme', 'business', 'main', 'street', 'other'] + [str(i) for i in range(300)]
            (base / 'vocab.txt').write_text('\n'.join(vocabulary), encoding='utf-8')
            tokenizer = BertTokenizer(vocab_file=str(base / 'vocab.txt'), do_lower_case=True)
            tokenizer.save_pretrained(base)
            BertForSequenceClassification(BertConfig(vocab_size=len(vocabulary), hidden_size=16, num_hidden_layers=1,
                num_attention_heads=2, intermediate_size=32, num_labels=1, max_position_embeddings=256)).save_pretrained(base)
            rng = np.random.default_rng(23)
            embeddings = unit(rng.normal(size=(300, 16)))
            (work / 'emb').mkdir()
            for split in ('train', 'test'):
                (dataset / split).mkdir(parents=True)
                for side in (1, 2, 3):
                    n = 240 if side == 1 else 300
                    records = pl.DataFrame({'entity_id': [f'S{side}-{i}' for i in range(n)],
                        'business_name': [f'acme business {i}' for i in range(n)],
                        'business_address': [f'{i} main street' if i % 4 else '' for i in range(n)],
                        'country': [('France' if split == 'test' and i % 3 == 0 else 'India' if i % 2 else 'US') for i in range(n)]})
                    records.write_csv(dataset / split / f'{split}_source{side}.tsv', separator='\t')
                    records.with_row_index('idx').with_columns(name_n=pl.col('business_name'), core_n=pl.col('business_name'),
                        addr_n=pl.col('business_address')).write_parquet(work / 'norm' / f'{split}_source{side}.parquet')
                np.save(work / 'emb' / f'{split}_s1.npy', embeddings[:240].astype(np.float16))
                np.save(work / 'emb' / f'{split}_tg.npy', np.concatenate([embeddings, embeddings]).astype(np.float16))
            pl.DataFrame({'source1_entity_id': [f'S1-{i}' for i in range(240)],
                'matched_entity_ids': [f'S2-{i},S3-{i}' if i % 10 else '' for i in range(240)]}).write_csv(dataset / 'train/train_ground_truth.tsv', separator='\t')
            env = {**os.environ, 'POLARS_MAX_THREADS': '2', 'OMP_NUM_THREADS': '2', 'R10_NEURAL_DEVICE': 'cpu',
                   'TOKENIZERS_PARALLELISM': 'false', 'PYTHONPATH': str(ROOT / 'code/business_entity_resolution/src') + os.pathsep + os.environ.get('PYTHONPATH', '')}
            def run(module, *more):
                print('smoke stage:', module, ' '.join(map(str, more[:2])), flush=True)
                result = subprocess.run([sys.executable, '-m', 'er_v2.' + module, *map(str, more), '--work', str(work)],
                    env=env, capture_output=True, text=True, timeout=180)
                self.assertEqual(result.returncode, 0, module + '\n' + result.stdout[-2000:] + result.stderr[-5000:])
            for split in ('train', 'test'):
                run('run_block', '--split', split, '--candidate-prefix', 'key', '--top-k', 8, '--name-k', 2, '--address-k', 2, '--rescue-k', 2)
                run('r10_retrieval', 'block', '--split', split, '--k', 4, '--search-k', 8, '--audit-queries', 4, '--threads', 2)
                run('r10_retrieval', 'merge', '--split', split, '--k', 4)
                run('run_features', '--split', split, '--dataset', dataset, '--enhanced', '--workers', 2, '--shard-pairs', 700)
            runtime = ['--model-dir', model, '--device', 'cpu', '--threads', 2, '--batch-rows', 100]
            run('train', *runtime, '--dataset', dataset, '--neural', '--neural-rescue-k', 2, '--rounds', 8)
            out = root / 'output'
            run('predict', *runtime, '--output', out / 'reference2')
            run('stage3', *runtime, '--dataset', dataset, '--rounds', 8, '--split', 'train', '--support-anchors', 25)
            run('stage3', *runtime, '--split', 'test', '--support-anchors', 25, '--output', out / 'reference')
            from er_v2.neural import Embeddings
            emb = Embeddings(work, 'train')
            self.assertIsInstance(emb.tg, np.memmap)
            with patch.dict(os.environ, {'R10_NEURAL_DEVICE': 'cpu'}):
                np.testing.assert_allclose(emb.cos(np.array([0, 1]), np.array([0, 1])), [1, 1], atol=.002)
            for split in ('train', 'test'):
                run('r10_ce', 'tokens', '--split', split, '--base-model', base, '--max-length', 32)
            sample = samples(work, dataset, TRAIN_FOLDS, 100)
            self.assertTrue(set(sample.select(fold_expr())['fold']) <= set(TRAIN_FOLDS))
            run('r10_ce', 'train', '--dataset', dataset, '--base-model', base, '--device', 'cpu', '--threads', 2,
                '--max-length', 32, '--batch', 8, '--score-batch', 16, '--epochs', 1, '--train-businesses', 25, '--valid-businesses', 10)
            self.assertEqual(json.loads((work / 'ce_model/training.json').read_text())['train_folds'], TRAIN_FOLDS)
            for split in ('train', 'test'):
                run('r10_ce', 'score', '--split', split, '--device', 'cpu', '--score-batch', 32, '--threads', 2)
            final = ['--dataset', dataset, '--output', out, '--device', 'cpu', '--threads', 2, '--rounds', 8]
            for stage in ('fit', 'select', 'evaluate', 'inference'):
                run('r10', stage, *final)
            # Exercise both output branches regardless of the tiny-data gate outcome.
            selection_path = work / 'selection.json'
            selection = json.loads(selection_path.read_text())
            frozen = selection_path.read_bytes()
            run('r10', 'evaluate', *final)
            self.assertEqual(selection_path.read_bytes(), frozen)
            for choice in ('reference', 'r10'):
                selection['selected'] = choice
                selection_path.write_text(json.dumps(selection))
                run('r10', 'inference', *final)
                matches = pl.read_csv(out / 'matching_results.tsv', separator='\t', infer_schema=False)
                self.assertEqual(matches.height, 240)  # includes France and singletons
                validator = ROOT / 'student_resource/utils/validate_submission.py'
                if validator.exists():
                    check = subprocess.run([sys.executable, str(validator), '--matching', str(out / 'matching_results.tsv'),
                        '--candidate', str(out / 'candidate_pairs.tsv'), '--test-dir', str(dataset / 'test'), '--check-ids'],
                        capture_output=True, text=True, timeout=30)
                    self.assertEqual(check.returncode, 0, check.stdout + check.stderr)

if __name__ == '__main__':
    unittest.main()
