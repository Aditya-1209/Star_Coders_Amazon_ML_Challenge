#!/bin/bash
# Always stop compute, including after failure or an unsuccessful result upload.
set -uo pipefail
trap 'shutdown -h now' EXIT
cd /opt/r9/repo || exit 1
export AWS_PAGER=''
# Export is bounded; large caches stay on EBS for restart. Do not export raw data.
timeout 300 /bin/bash scripts/aws/export_results.sh
status=$?
printf 'Result export exit code: %s\n' "$status"
if [[ "$status" -ne 0 ]]; then
  printf 'Upload incomplete. Logs/caches remain on the stopped instance EBS volume.\n'
fi
exit "$status"
