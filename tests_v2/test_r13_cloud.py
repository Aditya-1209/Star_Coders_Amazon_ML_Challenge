"""Cloud profile and supervisor checks. No cloud API, CUDA or real shutdown."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

import polars as pl
import torch
from er_v2.prepare import normalize_frame, _norm_chunk
from er_v2.r10_ce import amp_dtype, score_logits

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
        self.assertIn('--fused-optimizer', commands['ce_train'][0])
        old = run_r10.parser().parse_args([])
        self.assertEqual((old.prepare_buffer_rows, old.ce_precision, old.ce_fused_optimizer), (100000, 'fp16', False))

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
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'scripts').mkdir()
            shutil.copyfile(ROOT / 'scripts/vm_r13_job.sh', root / 'scripts/vm_r13_job.sh')
            (root / '.venv-r13/bin').mkdir(parents=True)
            python = root / '.venv-r13/bin/python'
            python.write_text('#!/usr/bin/env bash\nexit "${TEST_RUN_EXIT:-0}"\n')
            python.chmod(0o755)
            sudo = root / 'sudo'
            sudo.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$TEST_SHUTDOWN_LOG"\n')
            sudo.chmod(0o755)
            log = root / 'shutdowns'
            env = {**os.environ, 'PATH': str(root) + os.pathsep + os.environ['PATH'],
                   'TEST_SHUTDOWN_LOG': str(log), 'R13_SHUTDOWN_ON_EXIT': '1'}
            for status, remaining, enabled in ((0, 3600, '1'), (3, 3600, '1'), (0, 120, '1'), (0, 3600, '0')):
                log.write_text('')
                env.update(TEST_RUN_EXIT=str(status), R13_STOP_EPOCH=str(int(time.time()) + remaining),
                           R13_SHUTDOWN_ON_EXIT=enabled)
                result = subprocess.run(['bash', str(root / 'scripts/vm_r13_job.sh')], env=env,
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, status, result.stderr)
                if remaining > 600 and enabled == '1':
                    self.assertEqual(log.read_text().splitlines(), ['-n shutdown -c', '-n shutdown -h +10'])
                else:
                    self.assertEqual(log.read_text(), '')


if __name__ == '__main__':
    unittest.main()
