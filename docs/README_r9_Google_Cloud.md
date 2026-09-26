# R9 on Google Cloud — branch `r9_collab`

**Prepared only: no Google resources, training, inference or runtime tests were
started while building this branch.** The GPU check and synthetic tests run first
when you explicitly launch the experiment. There is no measured Google Cloud
runtime or new accuracy result yet.

This guide targets **Google Cloud Compute Engine**. Google Colab notebook runtimes
are a different product; see the note at the end if that is what you intended.
The underlying accuracy experiment is R9: another cross-fitted graph expansion,
equal-business loss weighting, a small model/blend comparison and a validation
gate with baseline fallback. Its folds, thresholds and evaluation rules are
unchanged. See [the R9 methodology](README_r9_AWS.md#why-try-this-alongside-r8).

## Which TPU should we use?

**None for this pipeline. Choose an NVIDIA GPU.** R9 uses XGBoost's CUDA tree
training plus Polars/RapidFuzz CPU processing. It has no TPU backend. Choosing a
TPU runtime would not accelerate these models; making a TPU useful would require
a separate model implementation in a supported framework, such as JAX or
PyTorch/XLA. That is a different experiment from this Google Cloud migration.
[XGBoost GPU support](https://xgboost.readthedocs.io/en/stable/gpu/index.html),
[Cloud TPU frameworks](https://docs.cloud.google.com/tpu/docs/runtimes)

Start with:

| Setting | Configuration |
|---|---|
| Machine | **`g2-standard-16`** |
| CPU | 16 vCPUs |
| Host memory | 64 GB |
| Accelerator | **1 NVIDIA L4, 24 GB GPU memory** |
| Disk | **200 GB persistent `pd-balanced`** |
| Image | Google Deep Learning VM Base, Ubuntu 24.04 / Python 3.12 / CUDA 12.9 / NVIDIA 580 |
| Region / starting zone | `us-central1` / `us-central1-a`, subject to quota and capacity |
| Provisioning | Standard On-Demand |
| Time allowance | 12 hours from VM start; runner reserves one hour for setup/export |

[Google G2 specifications](https://docs.cloud.google.com/compute/docs/gpus),
[supported DLVM images](https://docs.cloud.google.com/deep-learning-vm/docs/images)

The launcher also accepts `g2-standard-32` (32 vCPUs, 128 GB, still one L4) for
additional host-memory headroom. It adjusts workers from 12 to 24. Start with
the 16-vCPU option and inspect memory/stage timings before paying for a larger VM.
More GPU hardware does not automatically speed up string matching or graph joins.

## What changed for Google Cloud

- Cloud Storage replaces S3 for input archives, progress, models and final output.
- A Google NVIDIA image supplies a matching Linux/CUDA environment. Dependencies
  stay pinned in a separate Python 3.12 virtual environment.
- Feature shards, graph tables and temporary indexes stay on the VM's persistent
  disk. They are not streamed through a Cloud Storage or Google Drive mount.
- One heavy stage at a time, 12 CPU workers, single-threaded BLAS, country-scoped
  feature construction, and batched prediction are retained from R9.
- Small progress reports/logs upload every 10 minutes. Models and generated TSVs
  upload at completion/failure. Raw data and large feature caches stay on disk.
- A **Compute Engine runtime limit** stops the VM even if guest setup/training
  fails. Normal completion/failure also triggers a bounded export and shutdown.
- Restarting the VM does not restart training automatically. Explicit resume uses
  the existing checked cache; incomplete stages restart from the beginning.
- Instance identity supplies credentials through a dedicated service account.
  No service-account JSON key or GitHub token is put on the VM.

The runtime limit is a time guard, not a dollar cap. A provider-enforced stop may
interrupt an export; the periodic reports and persistent disk are the recovery
paths. [Compute Engine runtime limits](https://docs.cloud.google.com/compute/docs/instances/limit-vm-runtime)

## 1. Billing, pricing and quota

**AWS credit cannot pay for Google Cloud or Google Colab.** Confirm you have a
Google Cloud budget or eligible Google credits before launching anything here.

Check the selected region's **total G2 VM hourly price**, which includes its GPU,
CPU and RAM, in the Compute Engine creation screen or
[Google's pricing calculator](https://cloud.google.com/products/calculator).
Disk, Cloud Storage, external IP and transfer costs are additional. Do not use
the standalone GPU rate as the total VM price. If you have $100 of eligible
Google credit, reserve at least $20 and compute your allowance as
`80 / total_VM_hourly_price`. This is an allowance, not a benchmarked run time.
[G2 pricing](https://cloud.google.com/products/compute/pricing/accelerator-optimized)

Google's non-billable Free Trial account does not allow GPUs on VMs or quota
increases. An eligible billing account and sufficient **regional L4/G2 CPU quota**
and **GPUs (all regions)** quota are needed. Upgrading billing can expose you to
charges beyond credits; this repository never changes your account plan.
Verify the remaining credit and eligibility with the account owner.
[Free Trial restrictions](https://docs.cloud.google.com/free/docs/free-cloud-features),
[GPU quotas](https://docs.cloud.google.com/compute/resource-usage#gpu_quota)

Set a billing budget alert. Alerts do not stop resources. This release uses
Standard provisioning because Spot interruption behavior has not been tested.

## 2. Prepare source and data

On the friend's computer, authenticate to this private GitHub repository and run:

```bash
git clone --branch r9_collab --single-branch https://github.com/Aditya-1209/Star_Coders_Amazon_ML_Challenge.git
cd Star_Coders_Amazon_ML_Challenge
git archive --format=tar.gz --output=r9-source.tar.gz HEAD
```

The archive contains committed source, not uncommitted changes or ignored data.
Use the organizer's original ZIP, including the official validator and all seven
dataset files. Do not upload training data to GitHub.

## 3. Prepare a private project workspace

Use **Google Cloud Shell** in the friend's project. Replace these values with
your real project and globally unique bucket name. The following commands create
the named project resources; they do not start a GPU VM yet.

```bash
R9_PROJECT=your-google-project-id
R9_BUCKET=your-unique-private-r9-bucket
R9_REGION=us-central1
R9_ZONE=us-central1-a
R9_SA="r9-runner@$R9_PROJECT.iam.gserviceaccount.com"

gcloud services enable compute.googleapis.com storage.googleapis.com iap.googleapis.com --project "$R9_PROJECT"
gcloud storage buckets create "gs://$R9_BUCKET" --project "$R9_PROJECT" \
  --location "$R9_REGION" --uniform-bucket-level-access --public-access-prevention
gcloud iam service-accounts create r9-runner --project "$R9_PROJECT" --display-name 'R9 GPU experiment'
gcloud storage buckets add-iam-policy-binding "gs://$R9_BUCKET" \
  --member "serviceAccount:$R9_SA" --role roles/storage.objectUser

gcloud compute networks create r9-network --project "$R9_PROJECT" --subnet-mode custom
gcloud compute networks subnets create r9-subnet --project "$R9_PROJECT" \
  --network r9-network --region "$R9_REGION" --range 10.99.0.0/24
gcloud compute firewall-rules create r9-allow-iap-ssh --project "$R9_PROJECT" \
  --network r9-network --direction INGRESS --action ALLOW --rules tcp:22 \
  --source-ranges 35.235.240.0/20 --target-service-accounts "$R9_SA"
```

Use an existing suitably configured bucket/network/service account if your team
already has them; don't run duplicate create commands. The dedicated bucket role
allows reading/writing/listing experiment objects, including replacing progress
reports. It does not grant project-wide storage or Compute Engine administration.
The VM has an ephemeral external IP for outbound downloads; SSH ingress is only
from IAP. No notebook/web-server ports are exposed.

The person launching needs VM creation, service-account `actAs`, image access,
and the permissions for the setup commands. For SSH with sudo through IAP they
need `roles/iap.tunnelResourceAccessor`, `roles/compute.osAdminLogin` and any
required service-account-user access. Have the project owner grant these; do
not download or share service-account private keys.

In **Cloud Storage → your bucket → Upload**, upload:

| Local file | Object path |
|---|---|
| `r9-source.tar.gz` | `r9/input/source.tar.gz` |
| Original organizer dataset ZIP | `r9/input/dataset.zip` |

Then in Cloud Shell:

```bash
gcloud storage cp "gs://$R9_BUCKET/r9/input/source.tar.gz" /tmp/r9-source.tar.gz
mkdir -p ~/r9-launch
tar -xzf /tmp/r9-source.tar.gz -C ~/r9-launch
cd ~/r9-launch
R9_SHA=$(sha256sum /tmp/r9-source.tar.gz | cut -d ' ' -f 1)
```

## 4. Review the launch, then explicitly create the VM

The default command is a **dry run**, printing configuration and the `gcloud`
command without any API calls:

```bash
python3 scripts/gcp/create_vm.py \
  --project "$R9_PROJECT" --bucket "$R9_BUCKET" --service-account "$R9_SA" \
  --subnet r9-subnet --zone "$R9_ZONE" --source-sha256 "$R9_SHA" \
  --name r9-experiment --run-id first-run --hours 12
```

After checking the displayed configuration, region price, billing and quota,
repeat that command with **`--create` appended**. That creates the paid GPU VM
and starts setup and training. This action was not performed when preparing the
branch. No TPU resource is created.

The launcher uses the official image family
`common-cu129-ubuntu-2404-nvidia-580` in `deeplearning-platform-release`, one L4,
200 GB `pd-balanced`, `--max-run-duration=12h`, and
`--instance-termination-action=STOP`. The VM remains available with its persistent
disk when stopped. If the selected zone lacks capacity, use another zone with
G2 support and a compatible regional subnet; a quota does not reserve capacity.

The source hash is checked before extraction. First boot installs dependencies,
extracts data, checks CUDA and runs the test suite before the full experiment.
VM creation success does not mean setup or training has succeeded.

## 5. Progress, resume and results

From Cloud Shell:

```bash
gcloud compute ssh r9-experiment --project "$R9_PROJECT" --zone "$R9_ZONE" --tunnel-through-iap
```

Inside the VM:

```bash
sudo tail -n 60 /var/log/r9-bootstrap.log
sudo systemctl status r9.service --no-pager
sudo cat /opt/r9/repo/work/r9/run.json
sudo tail -n 60 /opt/r9/repo/work/service.log
```

`run.json` points to the current stage log in `work/r9/logs/`. Cloud Storage also
receives progress under `gs://YOUR-BUCKET/r9/results/first-run/`. You can close the
friend's laptop or browser; the VM service continues until completion, failure
or the configured time limit.

After a time limit, recheck your remaining budget and explicitly restart the
**same stopped VM**, reconnect, then resume the service:

```bash
# Cloud Shell:
gcloud compute instances start r9-experiment --project "$R9_PROJECT" --zone "$R9_ZONE"
gcloud compute ssh r9-experiment --project "$R9_PROJECT" --zone "$R9_ZONE" --tunnel-through-iap
# Inside the VM:
sudo systemctl start --no-block r9.service
```

The provider's runtime limit applies anew from that VM start. The service uses
`--resume` when a run state exists, validating code, dependencies, data and model
settings. It skips completed compatible stages. Interrupted fits/graph stages
restart from their beginning; there is no in-stage checkpoint. Don't change
source or dependencies midway or start a duplicate VM on the same result prefix.

On completion, the result prefix contains:

- `work/result.md`, `work/result.json`: measured local metrics, timings, settings,
  selection and output hashes; no claimed Amazon/France score.
- `work/selection.json`: tuning trials, acceptance gate and chosen model.
- `work/logs/validate.log`: the official format/ID validation log.
- `output/matching_results.tsv` and `output/candidate_pairs.tsv`: selected files.
- `work/models/` and `work/base/models/`: saved models/metadata for reference.

Only use submission files when `result.json` says `status: complete` and
`official_validation: PASS`. A rejected R9 proposal yields the rebuilt baseline.
Local macro F0.5 is not Amazon accuracy; fold 4 is report-only and was already
observed in earlier R7 work. No submission is made automatically.

If setup/tests fail, inspect the error before another run. A failed export leaves
the files on the persistent disk. Progress export errors appear in
`sudo journalctl -u r9-export.service`; they do not invalidate a completed model.
Full feature caches are not uploaded: keep the stopped disk if you need resume.
For an early setup failure before any upload, read the serial bootstrap log from
Cloud Shell without restarting compute:

```bash
gcloud compute instances get-serial-port-output r9-experiment --project "$R9_PROJECT" --zone "$R9_ZONE"
```

For the optional country-transfer diagnostic, use a **new VM/name/run ID** with
`--exclude-country India` or `--exclude-country US`. It rebuilds its own clean
training workspace and stops after evaluation. Run it only after checking the
first run's timings and remaining budget, not concurrently.

## Cleanup and Colab distinction

After downloading results and deciding you no longer need caches, delete the VM
and its boot disk from Compute Engine. **Deleting that disk removes the resume
cache.** Also remove unneeded Cloud Storage artifacts and any experiment-only
network/service-account resources. Stopped VMs still retain billable disk storage;
the bucket and stored objects remain after VM deletion.

Google Colab is a notebook service with different hardware availability, RAM,
session limits and billing. This branch's `scripts/gcp/*` uses Compute Engine
and systemd; it is not a hosted-Colab notebook installer. Selecting a TPU in
Colab still does not make XGBoost use that TPU. A full-data Colab adaptation would
need GPU selection, enough host RAM, local scratch storage and explicit durable
checkpoint handling; Google Drive should not be the active feature-cache disk.
