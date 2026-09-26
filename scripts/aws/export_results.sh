#!/bin/bash
set -euo pipefail
cd /opt/r9/repo
test -n "$R9_RESULTS"
aws s3 cp work/service.log "$R9_RESULTS/service.log"
aws s3 cp /var/log/r9-bootstrap.log "$R9_RESULTS/bootstrap.log"
aws s3 sync work/r9/ "$R9_RESULTS/work/" --exclude '*' --include '*.json' --include '*.md' --include '*.log'
# Only generated submissions; the same data already stays private in the bucket.
if [[ -d output/r9 ]]; then
  aws s3 sync output/r9/ "$R9_RESULTS/output/" --exclude '*' --include '*.tsv'
fi
