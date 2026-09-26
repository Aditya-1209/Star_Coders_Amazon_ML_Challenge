#!/bin/bash
# Called by the CloudFormation user data as root, on an Ubuntu 24.04 DLAMI.
set -Eeuo pipefail
test "$(id -u)" -eq 0
test "$#" -eq 4
RESULT_URI=$1
WALL_HOURS=$2
PROXY_COUNTRY=$3
REGION=$4
[[ "$WALL_HOURS" =~ ^[0-9]+$ ]] && (( WALL_HOURS >= 2 && WALL_HOURS <= 24 ))
[[ "$PROXY_COUNTRY" =~ ^(none|India|US)$ ]]
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3.12-venv libgomp1
cd /opt/r9/repo
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r code/business_entity_resolution/requirements_v2.txt
.venv/bin/python scripts/prepare_r9_data.py /opt/r9/dataset.zip
mkdir -p work output
chown -R ubuntu:ubuntu /opt/r9
# Only machine-local nonsecret configuration; no access keys or GitHub tokens.
cat >/etc/r9.env <<EOF
R9_RESULTS=$RESULT_URI
R9_MAX_HOURS=$((WALL_HOURS - 1))
R9_PROXY_COUNTRY=$PROXY_COUNTRY
AWS_DEFAULT_REGION=$REGION
AWS_PAGER=
AWS_RETRY_MODE=standard
AWS_MAX_ATTEMPTS=3
EOF
chmod 644 /etc/r9.env
cat >/etc/systemd/system/r9.service <<EOF
[Unit]
Description=R9 sequential GPU accuracy experiment
Wants=network-online.target
After=network-online.target

[Service]
Type=exec
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/r9/repo
EnvironmentFile=/etc/r9.env
Environment=PYTHONUNBUFFERED=1
ExecStart=/bin/bash /opt/r9/repo/scripts/aws/run_job.sh
ExecStopPost=+/bin/bash /opt/r9/repo/scripts/aws/finish_job.sh
TimeoutStartSec=infinity
RuntimeMaxSec=$((WALL_HOURS * 3600))
TimeoutStopSec=360
KillMode=control-group
Restart=no
StandardOutput=append:/opt/r9/repo/work/service.log
StandardError=append:/opt/r9/repo/work/service.log
EOF
# Intentionally not enabled: reboot does NOT silently spend another run's budget.
systemctl daemon-reload
systemctl start --no-block r9.service
