#!/bin/bash
set -uo pipefail
trap 'shutdown -h now' EXIT
cd /opt/r9/repo || exit 1
# Stop a periodic export before the final export to avoid overwriting newer reports.
systemctl stop r9-export.timer r9-export.service
timeout 300 .venv/bin/python scripts/gcp/export_results.py --full
status=$?
printf 'Final Cloud Storage export exit code: %s\n' "$status"
if [[ "$status" -ne 0 ]]; then
  printf 'Files remain on the persistent disk. Resume the stopped VM to retrieve them.\n'
fi
exit "$status"
