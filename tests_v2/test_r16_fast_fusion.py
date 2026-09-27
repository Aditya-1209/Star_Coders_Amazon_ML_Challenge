"""Independent-partition, identity mapping, exact fallback and bounded runner tests."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl
from er_v2 import r16_fast_fusion as fusion, r10
from er_v2.decision import decide_country
from er_v2.folds import fold_expr
from er_v2.metrics import macro_f05, by_country
from er_v2.predict import write_lists

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_r16_fast_fusion as runner


def pairs(data):
    return pl.DataFrame(data).with_columns(pl.col('sidx', 'tidx').cast(pl.UInt32))


def fixture(root, imperfect=False):
    root = root.resolve()
    base, work, out, exported, donor, data = [root / x for x in ('base', 'work', 'out', 'r12_export', 'donor', 'data')]
    for p in (base / 'norm', work, exported, donor): p.mkdir(parents=True)
    n = 3000
    for split in ('train', 'test'):
        (data / split).mkdir(parents=True)
        for side in (1, 2, 3):
            rec = pl.DataFrame({'entity_id': [f'S{side}-{i}' for i in range(n)],
                'business_name': [f'Business {i}' for i in range(n)], 'business_address': ['' for i in range(n)],
                'country': [('France' if split == 'test' and i % 3 == 0 else 'India' if i % 2 else 'US') for i in range(n)]})
            rec.write_csv(data / split / f'{split}_source{side}.tsv', separator='\t')
            rec.with_row_index('idx').write_parquet(base / 'norm' / f'{split}_source{side}.parquet')
    truth = pl.DataFrame({'source1_entity_id': [f'S1-{i}' for i in range(n)],
        'matched_entity_ids': [f'S2-{i}' if i % 10 else '' for i in range(n)]})
    truth.write_csv(data / 'train/train_ground_truth.tsv', separator='\t')
    positive = pairs({'sidx': np.arange(n, dtype=np.uint32), 'tidx': np.arange(n, dtype=np.uint32)})
    positive = positive.with_columns(score=pl.when(pl.col('sidx') % 10 == 0).then(.05)
        .when((pl.col('sidx') % 7 == 0) & pl.lit(imperfect)).then(.5).otherwise(.9).cast(pl.Float32))
    negative = positive.with_columns(tidx=pl.col('tidx') + n, score=pl.lit(.02, pl.Float32))
    scores = pl.concat([positive, negative]).with_columns(sid=pl.format('S1-{}', pl.col('sidx')),
        tid=pl.when(pl.col('tidx') < n).then(pl.format('S2-{}', pl.col('tidx'))).otherwise(pl.format('S3-{}', pl.col('tidx') - n)))
    meta = {'train_source1_ids': str(base / 'norm/train_source1.parquet'), 'model_id': 'synthetic-test-only', 'producer_note': 'Synthetic probabilities; no real R14 evidence', 'score_stage': 'pre_fusion', 'components': {}}
    for c in fusion.COMPONENTS:
        meta['components'][c] = {'score_kind': 'probability', 'score_column': 'score', 'fit_folds': [6, 7] if c == 'graph' else [0, 1, 8],
            'selection_folds': [3] if c == 'graph' else [9], 'files': {}}
    for stem, fold in (('fold3', 3), ('fold4', 4), ('test', None)):
        rows = scores.filter(fold_expr() == fold) if fold is not None else scores
        rows.write_parquet(exported / f'r12_{stem}.parquet')
        for c in fusion.COMPONENTS:
            name = f'{c}_{stem}.parquet'
            predicted = rows.select('sid', 'tid', 'score') if c == 'graph' else rows.select('sid', 'tid',
                score=pl.when((pl.col('tidx') < n) & (pl.col('sidx') % 10 != 0)).then(.98).otherwise(.01))
            predicted.reverse().write_parquet(donor / name)
            meta['components'][c]['files'][stem] = name
    provenance = donor / 'inputs.json'
    provenance.write_text(json.dumps(meta))
    args = SimpleNamespace(base_work=base, work=work, output=out, dataset=data, r12_export=exported,
        r14_provenance=provenance, device='cpu', threads=2, rounds=12)
    countries = r10.anchors(SimpleNamespace(work=base))
    target = r10.truth(fusion.parent(args), [3])
    cut = {'threshold': .7, 'country_thresholds': {}}
    old = {'version': 'r10-ce-ann-1', 'selected': 'r10', 'proposal': cut}
    a = countries.filter((fold_expr() == 3) & (r10.half() == 0))
    b = countries.filter((fold_expr() == 3) & (r10.half() == 1))
    def decision(country):
        return decide_country(scores.join(country.select('sidx'), on='sidx', how='semi'), .7, country, {}, 'score')
    cut['fold3A'] = macro_f05(decision(a), target, a['sidx'])['macro_f05']
    old['gate'] = {'candidate_by_country': by_country(decision(b), target, b)}
    c4 = countries.filter(fold_expr() == 4)
    measured = macro_f05(decision(c4), r10.truth(fusion.parent(args), [4]), c4['sidx'])
    for folder in (base, exported):
        (folder / 'selection.json').write_text(json.dumps(old))
        (folder / 'metrics.json').write_text(json.dumps({'local_fold4': measured}))
    anchors, targets = fusion.maps(args, 'test')
    s1 = anchors.rename({'sidx': 'idx'})
    frozen = root / 'frozen'
    frozen.mkdir()
    for name, rows, column in [('matching_results.tsv', decide_country(scores, .7, anchors.select('sidx', 'country'), {}, 'score'), 'matched_entity_ids'),
                               ('candidate_pairs.tsv', scores, 'candidate_entity_ids')]:
        write_lists(frozen / name, s1, rows, targets['entity_id'], column)
    (base / 'result.json').write_text(json.dumps({'output_sha256': {name: runner.file_hash(frozen / name) for name in ('matching_results.tsv', 'candidate_pairs.tsv')}}))
    return args


class FusionTests(unittest.TestCase):
    def test_union_keeps_missing_evidence_and_weight_zero_exact(self):
        base = pairs({'sidx': [0, 1], 'tidx': [0, 1], 'score': [.9, .8]})
        other = pairs({'sidx': [1, 2], 'tidx': [1, 2], 'graph': [.7, .6], 'ce_a': [.4, .6], 'ce_a_swapped': [.5, .7], 'ce_b': [.6, None]})
        f = fusion.feature_frame(base, other).with_columns(fusion=pl.Series([.1, .4, .6]))
        self.assertTrue(fusion.mixture(f, 0).equals(base))
        np.testing.assert_allclose(fusion.mixture(f, .5)['score'], [.9, .6, .6])
        self.assertIsNone(f['ce_a'][0])
        self.assertEqual(f['ce_a_available'][0], 0.)
        self.assertFalse(set(fusion.feature_names()) & {'label', 'sidx', 'tidx', 'country', 'fold'})

    def test_entity_ids_override_donor_indices_and_original_mapping_is_required(self):
        anchors = pl.DataFrame({'sidx': [0, 1], 'entity_id': ['a', 'b'], 'country': ['US', 'US']})
        targets = pl.DataFrame({'tidx': [0, 1], 'entity_id': ['x', 'y'], 'country': ['US', 'US']})
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / 'scores.parquet'
            pl.DataFrame({'sidx': [0], 'tidx': [0], 'score': [.8]}).write_parquet(p)
            with self.assertRaisesRegex(ValueError, 'original R14 norm'):
                fusion.checked_scores(p, anchors, targets)
            result = fusion.checked_scores(p, anchors, targets, donor_ids=(pl.Series(['b', 'a']), pl.Series(['y', 'x'])))
            self.assertEqual(result.select('sidx', 'tidx').row(0), (1, 1))
            pl.DataFrame({'sidx': [999], 'tidx': [999], 'sid': ['a'], 'tid': ['x'], 'score': [.8]}).write_parquet(p)
            self.assertEqual(fusion.checked_scores(p, anchors, targets).select('sidx', 'tidx').row(0), (0, 0))
            with self.assertRaises(ValueError): fusion.checked_scores(p, anchors, targets, baseline=True)
            for rows in ({'sid': ['bad'], 'tid': ['x'], 'score': [.8]}, {'sid': ['a', 'a'], 'tid': ['x', 'x'], 'score': [.8, .9]},
                         {'sid': ['a'], 'tid': ['x'], 'score': [float('nan')]}, {'sid': ['a'], 'tid': ['x'], 'score': [1.1]}):
                pl.DataFrame(rows).write_parquet(p)
                with self.assertRaises(ValueError): fusion.checked_scores(p, anchors, targets)

    def test_optimizer_rejects_gate_and_holdout_businesses(self):
        rows = pairs({'sidx': np.arange(1000, dtype=np.uint32), 'tidx': np.arange(1000, dtype=np.uint32), 'score': np.ones(1000)})
        with self.assertRaisesRegex(ValueError, 'only frozen R12 3A'):
            fusion.fit_fusion(rows, rows.select('sidx', 'tidx'), SimpleNamespace())

    def test_real_tree_selection_fallback_and_official_tsv_validation(self):
        with TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            before = runner.fingerprint([args.base_work, args.r12_export, args.r14_provenance.parent])
            real_fit = fusion.fit_fusion
            def fit(a, t, args):
                self.assertTrue(a.select((fold_expr() == 3) & (r10.half() == 0)).to_series().all())
                self.assertTrue(t.select((fold_expr() == 3) & (r10.half() == 0)).to_series().all())
                return real_fit(a, t, args)
            with patch.object(fusion, 'fit_fusion', side_effect=fit), patch.object(fusion.r10, 'gate', wraps=r10.gate) as gate, patch('builtins.print'):
                fusion.select(args)
                self.assertEqual(gate.call_count, 1)
                self.assertEqual(gate.call_args.args[-1], .0005)
                self.assertTrue(gate.call_args.args[-2].select(r10.half()).to_series().eq(1).all())
            frozen = (args.work / 'selection.json').read_bytes()
            self.assertEqual(json.loads(frozen)['selected'], 'r12')
            fusion.evaluate(args)
            fusion.inference(args)
            self.assertEqual((args.work / 'selection.json').read_bytes(), frozen)
            self.assertEqual(before, runner.fingerprint([args.base_work, args.r12_export, args.r14_provenance.parent]))
            self.assertTrue(json.loads((args.work / 'inference.json').read_text())['exact_r12_fallback'])
            validator = ROOT / 'student_resource/utils/validate_submission.py'
            if validator.exists():
                result = subprocess.run([sys.executable, str(validator), '--matching', str(args.output / 'matching_results.tsv'),
                    '--candidate', str(args.output / 'candidate_pairs.tsv'), '--test-dir', str(args.dataset / 'test'), '--check-ids'], capture_output=True, text=True, timeout=40)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            # Force the accepted code path only to verify artifacts, NOT as evidence of a real gain.
            selection = json.loads(frozen)
            selection.update(selected='r16_fast_ce_fusion', proposal={'threshold': .7, 'country_thresholds': {}, 'fusion_weight': .5})
            (args.work / 'selection.json').write_text(json.dumps(selection))
            fusion.evaluate(args)
            fusion.inference(args)
            self.assertEqual(pl.read_csv(args.output / 'matching_results.tsv', separator='\t', infer_schema=False).height, 3000)
            self.assertFalse(json.loads((args.work / 'inference.json').read_text())['exact_r12_fallback'])
            if validator.exists():
                result = subprocess.run([sys.executable, str(validator), '--matching', str(args.output / 'matching_results.tsv'),
                    '--candidate', str(args.output / 'candidate_pairs.tsv'), '--test-dir', str(args.dataset / 'test'), '--check-ids'], capture_output=True, text=True, timeout=40)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_select_does_not_read_fold4_before_saving_choice(self):
        with TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            for path in args.r14_provenance.parent.glob('*fold4.parquet'): path.unlink()
            (args.r12_export / 'r12_fold4.parquet').unlink()
            with patch('builtins.print'): fusion.select(args)
            self.assertTrue((args.work / 'selection.json').exists())

    def test_donor_training_fold_identity_is_verified_by_entity_id(self):
        with TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            meta = json.loads(args.r14_provenance.read_text())
            runner.check_fold_mapping(args, meta)
            reversed_ids = args.r14_provenance.parent / 'wrong_order.parquet'
            pl.read_parquet(meta['train_source1_ids'], columns=['entity_id']).reverse().write_parquet(reversed_ids)
            meta['train_source1_ids'] = str(reversed_ids)
            with self.assertRaisesRegex(ValueError, 'training folds differ'):
                runner.check_fold_mapping(args, meta)

    def test_provenance_rejects_final_fusion_and_gate_trained_ce(self):
        with TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            meta = json.loads(args.r14_provenance.read_text())
            runner.donor_files(args.r14_provenance)
            meta['score_stage'] = 'cross_fitted_final'
            args.r14_provenance.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'PRE-fusion'): runner.donor_files(args.r14_provenance)
            meta['score_stage'] = 'pre_fusion'
            meta['components']['ce_b']['fit_folds'] = [0, 1, 3, 8]
            args.r14_provenance.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'incompatible'): runner.donor_files(args.r14_provenance)

    def test_plan_deadline_and_isolation(self):
        args = runner.parser().parse_args(['--base-work', '/base', '--r12-export', '/export', '--r14-provenance', '/donor/inputs.json', '--dataset', '/data', '--validator', '/validate.py'])
        plan = runner.commands(args)
        self.assertEqual(list(plan), ['select', 'evaluate', 'inference', 'validate'])
        self.assertIn('--check-ids', plan['validate'][0])
        self.assertNotIn('er_v2.r10_ce', str(plan))
        args.shutdown_at = '2026-09-27T17:44:11Z'
        self.assertEqual(runner.remaining_seconds(args, datetime(2026, 9, 27, 17, 0, tzinfo=timezone.utc)), 2051)
        with self.assertRaises(TimeoutError): runner.remaining_seconds(args, datetime(2026, 9, 27, 17, 35, tzinfo=timezone.utc))
        args.work = args.base_work / 'new'
        with self.assertRaises(ValueError): runner.safe_paths(args)

    def test_real_runner_accepted_model_validates_resumes_and_rejects_changed_outputs(self):
        with TemporaryDirectory() as tmp:
            args = fixture(Path(tmp), imperfect=True)
            result_path = args.base_work / 'result.json'
            result = json.loads(result_path.read_text())
            result.update(status='complete', official_validation='PASS')
            result_path.write_text(json.dumps(result))
            validator = ROOT / 'student_resource/utils/validate_submission.py'
            if not validator.exists():
                self.skipTest('Organizer validator not supplied locally')
            identity = {'versions': {'polars': pl.__version__},
                        'data': runner.fingerprint([args.dataset / 'train', args.dataset / 'test']),
                        'validator': runner.fingerprint([validator])}
            (args.base_work / 'run.json').write_text(json.dumps({'status': 'complete', 'identity': identity}))
            argv = ['run_r16_fast_fusion.py']
            for key in ('base_work', 'r12_export', 'r14_provenance', 'dataset', 'work', 'output'):
                argv += ['--' + key.replace('_', '-'), str(getattr(args, key))]
            argv += ['--validator', str(validator), '--device', 'cpu', '--rounds', '100', '--threads', '2',
                     '--reserve-gb', '1', '--shutdown-at', '2099-01-01T00:00:00Z']
            original = runner.fingerprint([args.base_work])
            with patch.object(sys, 'argv', argv), patch('builtins.print'):
                runner.main()
            result = json.loads((args.work / 'result.json').read_text())
            self.assertEqual(result['official_validation'], 'PASS')
            selection = json.loads((args.work / 'selection.json').read_text())
            self.assertEqual(selection['selected'], 'r16_fast_ce_fusion')
            self.assertTrue(selection['gate']['passed'])
            self.assertEqual(selection['gate']['minimum_gain'], .0005)
            self.assertEqual(original, runner.fingerprint([args.base_work]))
            preserved = runner.fingerprint([args.work / 'fusion.json', args.output])
            with patch.object(sys, 'argv', argv + ['--resume']), patch('builtins.print'):
                runner.main()
            self.assertEqual(preserved, runner.fingerprint([args.work / 'fusion.json', args.output]))
            with (args.output / 'matching_results.tsv').open('a') as f:
                f.write('changed')
            with patch.object(sys, 'argv', argv + ['--resume']), patch('builtins.print'):
                with self.assertRaisesRegex(ValueError, 'Completed stage changed'):
                    runner.main()
            self.assertEqual(json.loads((args.work / 'run.json').read_text())['status'], 'failed')

    def test_wall_deadline_terminates_the_running_stage(self):
        with TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            argv = ['run_r16_fast_fusion.py']
            for key in ('base_work', 'r12_export', 'r14_provenance', 'dataset', 'work', 'output'):
                argv += ['--' + key.replace('_', '-'), str(getattr(args, key))]
            argv += ['--validator', str(ROOT / 'student_resource/utils/validate_submission.py'), '--reserve-gb', '1']
            pid_path = args.work / 'child.pid'
            child = 'import os,time;from pathlib import Path;Path(' + repr(str(pid_path)) + ').write_text(str(os.getpid()));time.sleep(60)'
            inputs = [args.base_work / 'selection.json']
            with patch.object(sys, 'argv', argv), patch.object(runner, 'preflight', return_value=({}, inputs, runner.fingerprint(inputs))), \
                 patch.object(runner, 'remaining_seconds', return_value=.2), patch.object(runner, 'commands', return_value={'select': ([sys.executable, '-c', child], [])}), patch('builtins.print'):
                with self.assertRaises(TimeoutError): runner.main()
            pid = int(pid_path.read_text())
            with self.assertRaises(ProcessLookupError): os.kill(pid, 0)
            self.assertEqual(json.loads((args.work / 'run.json').read_text())['status'], 'failed')

    def test_expired_shutdown_prevents_any_data_or_model_startup(self):
        args = runner.parser().parse_args(['--base-work', '/base', '--r12-export', '/export', '--r14-provenance', '/donor/inputs.json',
            '--dataset', '/data', '--validator', '/validate.py', '--shutdown-at', '2020-01-01T00:00:00Z'])
        with patch.object(runner, 'input_paths', side_effect=AssertionError('should not read')):
            with self.assertRaises(TimeoutError): runner.preflight(args)

    def test_launcher_preserves_venv_path_and_forwards_fast_command(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            py = root / 'fake-python'
            capture = root / 'args.json'
            py.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE"\n')
            py.chmod(0o755)
            env = {**os.environ, 'R16_PYTHON': str(py), 'R16_SHUTDOWN_AT': '2026-09-27T17:44:11Z',
                   'R16_R14_PROVENANCE': '/existing/raw.json', 'CAPTURE': str(capture)}
            result = subprocess.run(['bash', str(ROOT / 'scripts/vm_r16_fast_fusion.sh'), '/resource', '--preflight'], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            argv = capture.read_text().splitlines()
            self.assertEqual(argv[:2], ['-u', 'scripts/run_r16_fast_fusion.py'])
            self.assertIn('--preflight', argv)
            self.assertIn('/resource/dataset', argv)
            self.assertNotIn('pip', argv)


if __name__ == '__main__': unittest.main()
