"""Cloud profile and supervisor checks. No cloud API, CUDA or real shutdown."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import polars as pl
import torch
from er_v2.prepare import normalize_frame, _norm_chunk
from er_v2.r10_ce import PairTokens, amp_dtype, score_logits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_r10
import run_r13


class CloudTests(unittest.TestCase):
    def test_vm_profile_dispatches_thirty_normalization_chunks_and_bf16_ce(self):
        args = run_r13.parser(['--profile', 'vm']).parse_args(['--profile', 'vm'])
        self.assertEqual(args.prepare_buffer_rows // 25000, args.threads)
        self.assertEqual(args.ce_precision, 'bf16')
        self.assertTrue(args.ce_fused_optimizer)
        commands = run_r10.commands(args)
        prepare = commands['prepare'][0]
        self.assertEqual(prepare[prepare.index('--buffer-rows') + 1], '750000')
        for stage in ('ce_train', 'ce_score_train', 'ce_score_test'):
            command = commands[stage][0]
            self.assertEqual(command[command.index('--precision') + 1], 'bf16')
            self.assertEqual(float(command[command.index('--token-cache-gb') + 1]), 12)
        self.assertIn('--fused-optimizer', commands['ce_train'][0])
        old = run_r10.parser().parse_args([])
        self.assertEqual((old.prepare_buffer_rows, old.ce_precision, old.ce_fused_optimizer), (100000, 'fp16', False))
        self.assertEqual(old.ce_token_cache_gb, 0)
        self.assertEqual(run_r13.parser(['--profile', 'desktop']).parse_args(['--profile', 'desktop']).ce_token_cache_gb, 0)

    def test_ram_tokens_preserve_batches_and_fall_back_without_partial_copies(self):
        from transformers import BertTokenizer
        with TemporaryDirectory() as tmp:
            work = Path(tmp)
            root = work / 'ce_tokens'
            root.mkdir()
            vocab = root / 'vocab.txt'
            vocab.write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'acme', 'main']))
            BertTokenizer(vocab_file=str(vocab)).save_pretrained(root / 'tokenizer')
            for side in ('s1', 'tg'):
                np.save(root / f'train_{side}.npy', np.array([[5, 3, 6, 0], [6, 3, 0, 0]], np.uint32))
                np.save(root / f'train_{side}_lengths.npy', np.array([3, 2], np.uint16))
            pairs = pl.DataFrame({'sidx': [1, 0, 1], 'tidx': [0, 1, 1]})
            disk = PairTokens(work, 'train')
            budget = disk.cache_bytes / 1024**3
            ram = PairTokens(work, 'train', budget)
            self.assertEqual(ram.cache_mode, 'ram')
            self.assertTrue(all(not isinstance(a, np.memmap) and not a.flags.writeable
                                for a in (*ram.arrays.values(), *ram.lengths.values())))
            for swap in (False, True):
                for drops in (False, [True, False, True]):
                    expected = disk.batch(pairs, 'cpu', swap=swap, drop_address=drops)
                    actual = ram.batch(pairs, 'cpu', swap=swap, drop_address=drops)
                    for key in expected:
                        self.assertTrue(torch.equal(actual[key], expected[key]))
            np.testing.assert_array_equal(ram.pair_lengths(pairs), disk.pair_lengths(pairs))
            # Oversized cache does not even attempt a RAM copy.
            with patch('er_v2.r10_ce.np.array', side_effect=AssertionError('unexpected copy')):
                oversized = PairTokens(work, 'train', budget / 2)
            self.assertEqual(oversized.cache_mode, 'mmap')
            real_array, copies = np.array, []
            def fail_second(*args, **kwargs):
                copies.append(1)
                if len(copies) == 2:
                    raise MemoryError('test allocation failure')
                return real_array(*args, **kwargs)
            with patch('er_v2.r10_ce.np.array', side_effect=fail_second):
                failed = PairTokens(work, 'train', budget)
            self.assertEqual(failed.cache_mode, 'mmap')
            self.assertTrue(all(isinstance(a, np.memmap)
                                for a in (*failed.arrays.values(), *failed.lengths.values())))
            self.assertTrue(torch.equal(failed.batch(pairs, 'cpu')['input_ids'], disk.batch(pairs, 'cpu')['input_ids']))
            for invalid in (-1, float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    PairTokens(work, 'train', invalid)
            # Release file handles before the temporary directory closes on Windows.
            del disk, ram, oversized, failed

    def test_low_disk_preflight_and_invalid_ram_budget_fail_before_training(self):
        args = run_r13.parser().parse_args([])
        with patch.object(run_r10.shutil, 'disk_usage', return_value=SimpleNamespace(free=19 * 1024**3)):
            with self.assertRaisesRegex(RuntimeError, 'below 20 GiB reserve'):
                run_r10.preflight(args)
        for value in ('-1', 'nan', 'inf'):
            with self.subTest(value=value), patch.object(run_r10, 'preflight', side_effect=AssertionError('unexpected preflight')):
                with self.assertRaises(SystemExit):
                    run_r10.main(['--ce-token-cache-gb', value, '--plan'])

    def test_normalization_window_changes_parallelism_without_changing_records(self):
        class Pool:
            def __init__(self): self.sizes = []
            def imap(self, function, jobs):
                self.sizes.append(len(jobs))
                return map(function, jobs)
        rows = pl.DataFrame({'business_name': ['Acme, Ltd.', 'श्री गणेश', 'Café'] * 40,
                             'business_address': ['12 Main St', '', 'Côte rue 3'] * 40})
        small, large = Pool(), Pool()
        expected = normalize_frame(rows, small, chunk=2, buffer_rows=8)
        actual = normalize_frame(rows, large, chunk=2, buffer_rows=60)
        self.assertTrue(actual.equals(expected))
        self.assertEqual(max(small.sizes), 4)
        self.assertEqual(max(large.sizes), 30)

    def test_precision_checks_capability_and_keeps_cpu_in_float32(self):
        self.assertEqual(amp_dtype('cpu', 'bf16'), torch.float32)
        with patch('torch.cuda.is_bf16_supported', return_value=True):
            self.assertEqual(amp_dtype('cuda', 'bf16'), torch.bfloat16)
            self.assertEqual(amp_dtype('cuda', 'auto'), torch.bfloat16)
        with patch('torch.cuda.is_bf16_supported', return_value=False):
            self.assertEqual(amp_dtype('cuda', 'auto'), torch.float16)
            self.assertEqual(amp_dtype('cuda', 'fp16'), torch.float16)
            with self.assertRaisesRegex(RuntimeError, 'not supported'):
                amp_dtype('cuda', 'bf16')

    @unittest.skipUnless(shutil.which('bash'), 'Requires bash')
    def test_supervisor_stops_early_preserves_failure_and_never_extends_deadline(self):
        self.check_supervisor('r13')

    @unittest.skipUnless(shutil.which('bash'), 'Requires bash')
    def test_r16_supervisor_preserves_exit_status_and_deadline(self):
        self.check_supervisor('r16')

    def check_supervisor(self, version):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'scripts').mkdir()
            shutil.copyfile(ROOT / f'scripts/vm_{version}_job.sh', root / f'scripts/vm_{version}_job.sh')
            (root / f'.venv-{version}/bin').mkdir(parents=True)
            python = root / f'.venv-{version}/bin/python'
            python.write_text('#!/usr/bin/env bash\nexit "${TEST_RUN_EXIT:-0}"\n')
            python.chmod(0o755)
            sudo = root / 'sudo'
            sudo.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$TEST_SHUTDOWN_LOG"\n')
            sudo.chmod(0o755)
            log = root / 'shutdowns'
            env = {**os.environ, 'PATH': str(root) + os.pathsep + os.environ['PATH'],
                   'TEST_SHUTDOWN_LOG': str(log)}
            for status, remaining, enabled in ((0, 3600, '1'), (3, 3600, '1'), (0, 120, '1'), (0, 3600, '0')):
                log.write_text('')
                env.update(TEST_RUN_EXIT=str(status), **{version.upper() + '_STOP_EPOCH': str(int(time.time()) + remaining),
                           version.upper() + '_SHUTDOWN_ON_EXIT': enabled})
                result = subprocess.run(['bash', str(root / f'scripts/vm_{version}_job.sh')], env=env,
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, status, result.stderr)
                if remaining > 600 and enabled == '1':
                    self.assertEqual(log.read_text().splitlines(), ['-n shutdown -c', '-n shutdown -h +10'])
                else:
                    self.assertEqual(log.read_text(), '')


if __name__ == '__main__':
    unittest.main()
