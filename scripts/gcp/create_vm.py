#!/usr/bin/env python3
"""Prepare a Google Cloud G2 launch. Default is dry-run; --create provisions a VM.

Run in Cloud Shell after uploading the source/dataset and granting the dedicated
VM service account access to the private bucket. This script never creates IAM
keys, changes billing plans, or starts TPU resources.
"""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
IMAGE_FAMILY = "common-cu129-ubuntu-2404-nvidia-580"
MACHINES = {"g2-standard-16": {"cpus": 16, "threads": 12},
            "g2-standard-32": {"cpus": 32, "threads": 24}}


def check(args):
    patterns = {"project": r"[a-z][a-z0-9-]{4,28}[a-z0-9]", "name": r"[a-z][a-z0-9-]{0,61}[a-z0-9]",
                "run_id": r"[a-zA-Z0-9-]{1,50}", "zone": r"[a-z]+-[a-z]+[0-9]+-[a-z]",
                "bucket": r"[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]", "source_sha256": r"[a-f0-9]{64}",
                "service_account": r"[a-z0-9-]+@[a-z0-9-]+\.iam\.gserviceaccount\.com"}
    for field, pattern in patterns.items():
        if not re.fullmatch(pattern, getattr(args, field)):
            raise ValueError(f"Invalid {field}")
    if not 2 <= args.hours <= 24 or not 150 <= args.disk_gb <= 500:
        raise ValueError("Use 2–24 hours and 150–500 GB of persistent disk")
    if not args.subnet or any(c.isspace() for c in args.subnet):
        raise ValueError("Specify an existing subnet name or resource URL")


def config(args):
    return {"project": args.project, "source_uri": f"gs://{args.bucket}/r9/input/source.tar.gz",
            "source_sha256": args.source_sha256, "dataset_uri": f"gs://{args.bucket}/r9/input/dataset.zip",
            "results_uri": f"gs://{args.bucket}/r9/results/{args.run_id}", "hours": args.hours,
            "threads": MACHINES[args.machine_type]["threads"], "exclude_country": args.exclude_country}


def command(args, config_file):
    return ["gcloud", "compute", "instances", "create", args.name,
            "--project", args.project, "--zone", args.zone, "--machine-type", args.machine_type,
            "--image-project", "deeplearning-platform-release", "--image-family", IMAGE_FAMILY,
            "--boot-disk-type", "pd-balanced", "--boot-disk-size", f"{args.disk_gb}GB",
            "--subnet", args.subnet, "--service-account", args.service_account, "--scopes", "cloud-platform",
            "--maintenance-policy", "TERMINATE", "--no-restart-on-failure", "--provisioning-model", "STANDARD",
            "--max-run-duration", f"{args.hours}h", "--instance-termination-action", "STOP",
            "--metadata", "enable-oslogin=TRUE,block-project-ssh-keys=TRUE,install-nvidia-driver=True",
            "--metadata-from-file", f"startup-script={ROOT / 'scripts/gcp/startup.sh'},r9-config={config_file}",
            "--labels", "application=r9,experiment=r9-collab"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", required=True)
    p.add_argument("--bucket", required=True)
    p.add_argument("--service-account", required=True)
    p.add_argument("--subnet", required=True)
    p.add_argument("--source-sha256", required=True)
    p.add_argument("--zone", default="us-central1-a")
    p.add_argument("--name", default="r9-experiment")
    p.add_argument("--run-id", default="first-run")
    p.add_argument("--machine-type", choices=list(MACHINES), default="g2-standard-16")
    p.add_argument("--hours", type=int, default=12)
    p.add_argument("--disk-gb", type=int, default=200)
    p.add_argument("--exclude-country", choices=["none", "India", "US"], default="none")
    p.add_argument("--create", action="store_true", help="Create the paid VM and start setup/training")
    args = p.parse_args()
    try:
        check(args)
    except ValueError as error:
        p.error(str(error))
    value = config(args)
    print(json.dumps(value, indent=2))
    if not args.create:
        print("DRY RUN: no API calls made. Add --create only after checking Google billing, pricing and quota.")
        print(shlex.join(command(args, "<temporary-config.json>")))
        return
    with tempfile.TemporaryDirectory(prefix="r9-launch-") as temporary:
        path = Path(temporary) / "config.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        subprocess.run(command(args, path), check=True)
    print(f"VM created. Training status is separate from VM creation. Results: {value['results_uri']}")


if __name__ == "__main__":
    main()
