"""Linux launcher regression checks; fake CUDA/install/shutdown, never start training."""
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == 'posix' and shutil.which('flock') and shutil.which('sha256sum'),
                     'Linux launcher requires bash/flock/sha256sum')
class VMTests(unittest.TestCase):
    version = 'r12'

    def fixture(self, root):
        (root / 'scripts').mkdir()
        shutil.copyfile(ROOT / f'scripts/vm_{self.version}.sh', root / f'scripts/vm_{self.version}.sh')
        if self.version == 'r13':
            shutil.copyfile(ROOT / 'scripts/vm_r13_job.sh', root / 'scripts/vm_r13_job.sh')
        requirements = root / 'code/business_entity_resolution'
        requirements.mkdir(parents=True)
        for name in ('requirements_v2.txt', 'requirements_r10.txt'):
            (requirements / name).write_text('# fixture\n')
        resource = root / 'organizer resources'
        for split in ('train', 'test'):
            (resource / 'dataset' / split).mkdir(parents=True)
            for side in (1, 2, 3):
                (resource / 'dataset' / split / f'{split}_source{side}.tsv').touch()
        (resource / 'dataset/train/train_ground_truth.tsv').touch()
        (resource / 'utils').mkdir()
        (resource / 'utils/validate_submission.py').touch()
        # Stale local data must not redirect the supplied organizer path.
        (root / 'student_resource').mkdir()
        fakebin = root / 'fakebin'
        fakebin.mkdir()
        sudo = fakebin / 'sudo'
        sudo.write_text('#!/usr/bin/env bash\nprintf "sudo %s\\n" "$*" >> "$R12_TEST_CALLS"\n')
        sudo.chmod(0o755)
        venv = root / f'.venv-{self.version}/bin'
        venv.mkdir(parents=True)
        python = venv / 'python'
        python.write_text('''#!/usr/bin/env bash
printf 'python %s\n' "$*" >> "$R12_TEST_CALLS"
case "$*" in
  *'pip check'*) exit "${R12_TEST_FAIL_CHECK:-0}" ;;
  *'math.ceil'*) echo 720 ;;
  *'--preflight'*|*'--plan'*|*'unittest'*|*'-m pip'*|*'import torch'*) exit 0 ;;
esac
echo running
touch "$R12_TEST_STARTED"
sleep 2
''')
        python.chmod(0o755)
        env = {**os.environ, 'PATH': str(fakebin) + os.pathsep + os.environ['PATH'],
               'R12_TEST_CALLS': str(root / 'calls'), 'R12_TEST_STARTED': str(root / 'started')}
        return resource, env

    def test_repairs_incomplete_env_preserves_data_path_and_prevents_duplicate_launch(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            resource, env = self.fixture(root)
            command = ['bash', str(root / f'scripts/vm_{self.version}.sh'), str(resource)]
            first = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            for _ in range(50):
                if (root / 'started').exists():
                    break
                time.sleep(.02)
            self.assertTrue((root / f'.venv-{self.version}/.ready').exists())
            calls = (root / 'calls').read_text()
            self.assertIn('pip check', calls)  # executable alone did not skip package repair
            self.assertIn('--dataset ' + str(resource / 'dataset'), calls)
            self.assertIn('--validator ' + str(resource / 'utils/validate_submission.py'), calls)
            self.assertIn('unittest discover -s tests_v2', calls)
            self.assertIn('sudo shutdown -h +720', calls)
            duplicate = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(duplicate.returncode, 0)
            self.assertIn('already active', duplicate.stdout)
            time.sleep(2.1)
            self.assertIn('running', (root / f'work/{self.version}/runner.log').read_text())

    def test_failed_dependency_install_never_marks_ready_or_launches(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            resource, env = self.fixture(root)
            env['R12_TEST_FAIL_CHECK'] = '1'
            result = subprocess.run(['bash', str(root / f'scripts/vm_{self.version}.sh'), str(resource)],
                                    env=env, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / f'.venv-{self.version}/.ready').exists())
            self.assertFalse((root / 'started').exists())
            self.assertNotIn('shutdown', (root / 'calls').read_text())


if __name__ == '__main__':
    unittest.main()
