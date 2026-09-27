#!/usr/bin/env bash
# Internal supervisor, launched by vm_r13.sh with its inherited run lock.
set -uo pipefail
cd "$(dirname "$0")/.."
status=0
.venv-r13/bin/python -u scripts/run_r13.py "$@" || status=$?
echo "R13 runner exited with status $status at $(date -u +%FT%TZ)"

# Keep an imminent overall deadline. An unknown deadline means this helper was
# called directly; do not schedule a shutdown outside the VM launcher contract.
remaining=$(( (${R13_STOP_EPOCH:-0} - $(date +%s) + 59) / 60 ))
if [ "${R13_SHUTDOWN_ON_EXIT:-0}" = 1 ] && [ "$remaining" -gt 10 ]; then
  if sudo -n shutdown -c && sudo -n shutdown -h +10; then
    echo 'VM shutdown scheduled in 10 minutes; outputs and logs remain on Persistent Disk.'
  else
    echo 'Could not schedule early shutdown; restoring the overall deadline.'
    sudo -n shutdown -h "+$remaining" || echo 'WARNING: check VM shutdown in Google Cloud Console.'
  fi
fi
exit "$status"
