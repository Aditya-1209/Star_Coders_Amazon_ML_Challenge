"""Run the R15 shell launcher with fake Python/shutdown, without touching a VM."""
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == 'posix', 'Linux launcher requires bash/flock')
class LauncherTests(unittest.TestCase):
    def fixture(self, root):
        (root / 'scripts').mkdir()
        shutil.copyfile(ROOT / 'scripts/vm_r15.sh', root / 'scripts/vm_r15.sh')
        (root / 'resource with spaces').mkdir()
        (root / '.venv-r12/bin').mkdir(parents=True)
        fake = root / 'fakebin'
        fake.mkdir()
        sudo = fake / 'sudo'
        sudo.write_text('#!/usr/bin/env bash\nprintf "sudo %s\\n" "$*" >> "$TEST_CALLS"\n')
        sudo.chmod(0o755)
        python = root / '.venv-r12/bin/python'
        python.write_text('''#!/usr/bin/env bash
printf 'python %s\n' "$*" >> "$TEST_CALLS"
case "$*" in
 *'--preflight'*) exit "${FAIL_PREFLIGHT:-0}" ;;
 *'math.ceil'*) echo 420 ;;
 *'--plan'*|*'unittest'*|*'-m pip'*) exit 0 ;;
esac
touch "$TEST_STARTED"
sleep 2
echo complete
''')
        python.chmod(0o755)
        return {**os.environ, 'PATH': str(fake) + os.pathsep + os.environ['PATH'],
                'TEST_CALLS': str(root / 'calls'), 'TEST_STARTED': str(root / 'started')}

    def test_preflight_failure_does_not_rearm_shutdown_or_launch(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = self.fixture(root)
            env['FAIL_PREFLIGHT'] = '1'
            result = subprocess.run(['bash', str(root / 'scripts/vm_r15.sh'), str(root / 'resource with spaces')],
                                    env=env, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn('shutdown', (root / 'calls').read_text())
            self.assertFalse((root / 'started').exists())

    def test_detached_launch_preserves_paths_and_prevents_duplicate(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = self.fixture(root)
            command = ['bash', str(root / 'scripts/vm_r15.sh'), str(root / 'resource with spaces')]
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for _ in range(100):
                if (root / 'started').exists():
                    break
                time.sleep(.01)
            self.assertTrue((root / 'started').exists())
            duplicate = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(duplicate.returncode, 0)
            self.assertIn('already active', duplicate.stdout)
            calls = (root / 'calls').read_text()
            self.assertIn('--base-work work/r12', calls)
            self.assertIn('--dataset ' + str(root / 'resource with spaces/dataset'), calls)
            self.assertIn('unittest discover -s tests_v2', calls)
            self.assertIn('shutdown -h +420', calls)
            time.sleep(2.1)
            self.assertIn('complete', (root / 'work/r15/runner.log').read_text())


if __name__ == '__main__':
    unittest.main()
