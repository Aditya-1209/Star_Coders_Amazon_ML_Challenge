#!/bin/bash
# Google runs startup scripts on every boot. Never launch another experiment on restart.
set -Eeuo pipefail
exec > >(tee -a /var/log/r9-bootstrap.log /dev/ttyS0) 2>&1
exec 9>/var/lock/r9-bootstrap.lock
flock -n 9 || exit 0
if [[ -f /opt/r9/setup-complete ]]; then
  printf 'Setup already completed. Explicitly start r9.service to resume.\n'
  exit 0
fi
fail() {
  status=$?
  trap - ERR
  set +e
  if [[ -f /etc/r9-gcp.env ]]; then
    source /etc/r9-gcp.env
    timeout 60 gcloud storage cp /var/log/r9-bootstrap.log "$R9_RESULTS/bootstrap.log"
  fi
  shutdown -h now
  exit "$status"
}
trap fail ERR
# A Compute Engine --max-run-duration deadline is already set outside this guest.
command -v gcloud
mkdir -p /opt/r9/repo
python3 - <<'PY'
import json, re, shlex, urllib.request
from pathlib import Path
request = urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/attributes/r9-config',
                                 headers={'Metadata-Flavor': 'Google'})
with urllib.request.urlopen(request, timeout=30) as response:
    config = json.load(response)
if not 2 <= config['hours'] <= 24 or not 1 <= config['threads'] <= 24:
    raise ValueError('Invalid runtime limits')
if config['exclude_country'] not in ('none', 'India', 'US'):
    raise ValueError('Invalid excluded country')
if not re.fullmatch('[a-f0-9]{64}', config['source_sha256']):
    raise ValueError('Invalid source digest')
for key in ('source_uri', 'dataset_uri', 'results_uri'):
    if not config[key].startswith('gs://') or '\n' in config[key]:
        raise ValueError('Invalid storage URI')
Path('/etc/r9-gcp.json').write_text(json.dumps(config, indent=2))
values = {'R9_SOURCE': config['source_uri'], 'R9_SOURCE_SHA256': config['source_sha256'],
          'R9_DATASET': config['dataset_uri'], 'R9_RESULTS': config['results_uri'],
          'R9_MAX_HOURS': config['hours'] - 1, 'R9_THREADS': config['threads'],
          'R9_PROXY_COUNTRY': config['exclude_country'], 'CLOUDSDK_CORE_PROJECT': config['project'],
          'CLOUDSDK_CORE_DISABLE_PROMPTS': '1'}
Path('/etc/r9-gcp.env').write_text(''.join(f'{key}={shlex.quote(str(value))}\n' for key, value in values.items()))
PY
set -a
source /etc/r9-gcp.env
set +a
gcloud storage cp "$R9_SOURCE" /opt/r9/source.tar.gz
printf '%s  /opt/r9/source.tar.gz\n' "$R9_SOURCE_SHA256" | sha256sum --check -
python3 - <<'PY'
import tarfile
with tarfile.open('/opt/r9/source.tar.gz') as archive:
    archive.extractall('/opt/r9/repo', filter='data')
PY
gcloud storage cp "$R9_DATASET" /opt/r9/dataset.zip
bash /opt/r9/repo/scripts/gcp/install_service.sh
trap - ERR
