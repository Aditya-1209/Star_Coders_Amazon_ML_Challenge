#!/bin/bash
set -Eeuo pipefail
test "$(id -u)" -eq 0
source /etc/r9-gcp.env
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3.12-venv libgomp1
cd /opt/r9/repo
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r code/business_entity_resolution/requirements_v2.txt
.venv/bin/python scripts/prepare_r9_data.py /opt/r9/dataset.zip
if ! id r9runner >/dev/null 2>&1; then
  useradd --system --create-home --home-dir /home/r9runner --shell /usr/sbin/nologin r9runner
fi
mkdir -p work output
chown -R r9runner:r9runner /opt/r9
cat >/etc/systemd/system/r9.service <<EOF
[Unit]
Description=R9 Google Cloud GPU experiment
Wants=network-online.target
After=network-online.target

[Service]
Type=exec
User=r9runner
Group=r9runner
WorkingDirectory=/opt/r9/repo
EnvironmentFile=/etc/r9-gcp.env
ExecStart=/bin/bash /opt/r9/repo/scripts/gcp/run_job.sh
ExecStopPost=+/bin/bash /opt/r9/repo/scripts/gcp/finish_job.sh
RuntimeMaxSec=$((R9_MAX_HOURS * 3600 + 600))
TimeoutStopSec=360
KillMode=control-group
Restart=no
StandardOutput=append:/opt/r9/repo/work/service.log
StandardError=append:/opt/r9/repo/work/service.log
EOF
cat >/etc/systemd/system/r9-export.service <<'EOF'
[Unit]
Description=Upload small R9 progress artifacts to private Cloud Storage
After=network-online.target
[Service]
Type=oneshot
EnvironmentFile=/etc/r9-gcp.env
WorkingDirectory=/opt/r9/repo
ExecStart=/usr/bin/timeout 120 /opt/r9/repo/.venv/bin/python scripts/gcp/export_results.py
EOF
cat >/etc/systemd/system/r9-export.timer <<'EOF'
[Unit]
Description=Upload R9 progress every 10 minutes
[Timer]
OnBootSec=10min
OnUnitActiveSec=10min
Unit=r9-export.service
[Install]
WantedBy=timers.target
EOF
systemctl daemon-reload
systemctl enable --now r9-export.timer
touch /opt/r9/setup-complete
# The experiment itself is not enabled at boot; resuming is an explicit action.
systemctl start --no-block r9.service
