"""Linux launcher tests: preserve R15 checkout/job and existing shutdown timer."""
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == 'posix', 'Linux launcher requires bash/flock')
class FastLauncherTests(unittest.TestCase):
    def fixture(self, root):
        (root / 'scripts').mkdir()
        shutil.copyfile(ROOT / 'scripts/vm_r15_rescue.sh', root / 'scripts/vm_r15_rescue.sh')
        (root / 'resource with spaces').mkdir()
        (root / 'old/work/r12').mkdir(parents=True)
        (root / 'old/.venv-r12/bin').mkdir(parents=True)
        fake = root / 'fakebin'
        fake.mkdir()
        sudo = fake / 'sudo'
        sudo.write_text('#!/usr/bin/env bash\nprintf "sudo %s\\n" "$*" >> "$TEST_CALLS"\n')
        sudo.chmod(0o755)
        python = root / 'old/.venv-r12/bin/python'
        python.write_text('''#!/usr/bin/env bash
printf 'python %s\n' "$*" >> "$TEST_CALLS"
case "$*" in
 *'R15/R16 is still running'*) exit "${FAIL_ACTIVE:-0}" ;;
 *'--preflight'*) exit "${FAIL_PREFLIGHT:-0}" ;;
 *'--plan'*|*'unittest'*|*'-m pip'*) exit 0 ;;
esac
touch "$TEST_STARTED"
sleep 2
echo complete
''')
        python.chmod(0o755)
        return {**os.environ, 'PATH': str(fake) + os.pathsep + os.environ['PATH'],
                'TEST_CALLS': str(root / 'calls'), 'TEST_STARTED': str(root / 'started'),
                'R15_RESCUE_BASE_WORK': str(root / 'old/work/r12'), 'R15_RESCUE_PYTHON': str(python)}

    def test_active_r15_and_bad_preflight_never_launch_or_change_timer(self):
        for failure in ('FAIL_ACTIVE', 'FAIL_PREFLIGHT'):
            with self.subTest(failure=failure), TemporaryDirectory() as tmp:
                root = Path(tmp)
                env = self.fixture(root)
                env[failure] = '1'
                result = subprocess.run(['bash', str(root / 'scripts/vm_r15_rescue.sh'), str(root / 'resource with spaces')],
                    env=env, capture_output=True, text=True, timeout=20)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('shutdown', (root / 'calls').read_text())
                self.assertFalse((root / 'started').exists())

    def test_detached_launch_preserves_old_environment_and_timer(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = self.fixture(root)
            command = ['bash', str(root / 'scripts/vm_r15_rescue.sh'), str(root / 'resource with spaces')]
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
            self.assertIn('--base-work ' + str(root / 'old/work/r12'), calls)
            self.assertIn('--dataset ' + str(root / 'resource with spaces/dataset'), calls)
            self.assertIn('unittest discover -s tests_v2', calls)
            self.assertIn('sudo shutdown --show', calls)
            self.assertNotIn('shutdown -c', calls)
            self.assertNotIn('shutdown -h', calls)
            time.sleep(2.1)


if __name__ == '__main__':
    unittest.main()
