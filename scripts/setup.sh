#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v python3.12 >/dev/null 2>&1; then
  er_python=python3.12
else
  er_python=python3
fi
"$er_python" -c 'import sys; assert sys.version_info[:2] == (3,12), "Use Python 3.12. On Windows, use Ubuntu 24.04 in WSL2."'
if [ ! -x .venv/bin/python ]; then
  "$er_python" -m venv .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3,12), "The existing .venv must use Python 3.12."'
.venv/bin/python -m pip install -r code/business_entity_resolution/requirements.txt
.venv/bin/python -c 'import sqlite3; c=sqlite3.connect(":memory:"); c.execute("CREATE VIRTUAL TABLE probe USING fts5(text)"); print("Python and SQLite FTS5 are ready.")'
PYTHONPATH=code/business_entity_resolution/src .venv/bin/python -m unittest discover -s code/business_entity_resolution/tests -v
.venv/bin/python -m unittest discover -s tests -v
echo 'Setup complete. See README.md for the one-command full run.'
