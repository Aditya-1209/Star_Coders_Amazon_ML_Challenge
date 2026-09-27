# Google Cloud setup for R13

Use **Compute Engine `g2-standard-32` on-demand** for the full run. This matches
R13's single-GPU execution and gives the CPU/Polars/XGBoost stages room to work.
It is a recommendation based on the pipeline, not a measured comparison of
full-run cost across GPU types.

| Console setting | Choose |
| --- | --- |
| Machine family | GPU / G2 |
| Machine type | `g2-standard-32` |
| GPU | 1 NVIDIA L4, 24 GB VRAM, included with G2 |
| vCPUs / RAM | 32 / 128 GB |
| Region / first zone | Iowa `us-central1` / `us-central1-a` |
| Provisioning model | Standard (on-demand) |
| OS | Ubuntu 24.04 LTS, x86-64, standard image |
| Boot/work disk | 300 GB SSD Persistent Disk (`pd-ssd`) |
| Maximum VM runtime / action | 12 hours / STOP |
| Automatic restart | Disabled |

G2 is also listed in Iowa zones `b` and `c`; availability still depends on
project quota and current capacity. Machine specifications and supported zones
are from [Google's machine documentation](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines)
and [GPU locations](https://docs.cloud.google.com/compute/docs/regions-zones/gpu-regions-zones).

`g2-standard-24` has **two** L4s, and R13 would leave one unused. The smaller
`g2-standard-16` has the same single GPU but half the host memory and CPU count;
the 128 GB profile is intended for the larger machine. An A100 is an option
after profiling shows GPU time dominates, but its higher hourly cost alone
does not establish a faster or cheaper end-to-end run.

## Cost

Google's Iowa pricing table, checked on **2026-09-27**, lists:

| Machine | On-demand USD/hour | Role in this pipeline |
| --- | ---: | --- |
| `g2-standard-16` | $1.1472 | Lower cost/hour, less CPU/RAM |
| **`g2-standard-32`** | **$1.7344** | Recommended full-run configuration |
| `g2-standard-24` | $2.0008 | Pays for an unused second GPU |
| `a2-highgpu-1g` | $3.6734 | 1 A100; 12 vCPUs / 85 GB RAM |

Source: [Google accelerator-optimized pricing](https://cloud.google.com/products/compute/pricing/accelerator-optimized),
Iowa, hourly, standard price column. A 12-hour `g2-standard-32` allocation is
about **$20.81 compute**, plus disk, network and tax. This is a budget example,
not a prediction that training will finish in 12 hours. Confirm the current
estimate in Console; prices and regional capacity can change. Retained disks
and reserved IPs can continue billing after the VM stops.

Use on-demand for the first complete experiment. The runner resumes completed
stages but restarts the interrupted active stage; a Spot eviction during
encoder/CE training can discard hours from that stage.

## Create the VM

From Cloud Shell, replace `YOUR_PROJECT`. **Running this command creates the
paid VM**; it does not upload data or start training.

```bash
gcloud compute instances create r13-l4 \
  --project=YOUR_PROJECT --zone=us-central1-a \
  --machine-type=g2-standard-32 \
  --image-project=ubuntu-os-cloud --image-family=ubuntu-2404-lts-amd64 \
  --boot-disk-type=pd-ssd --boot-disk-size=300GB \
  --no-boot-disk-auto-delete \
  --no-service-account --no-scopes \
  --provisioning-model=STANDARD --maintenance-policy=TERMINATE \
  --no-restart-on-failure \
  --max-run-duration=12h --instance-termination-action=STOP
```

The independent Compute Engine runtime limit also covers setup time and works
if the guest launcher crashes. [Google documents these runtime flags](https://docs.cloud.google.com/compute/docs/instances/limit-vm-runtime).
The command uses the project's default network; add `--subnet` if your project
requires a specific existing subnet. The VM needs outbound package/model
downloads and your usual SSH access. No VM service account or public web-server
ports are needed by the training pipeline.

Use the regular Ubuntu image. G2's supported-image constraints exclude Deep
Learning VM boot images, and Ubuntu's preinstalled accelerated images target
other GPU series; see [G2 limitations](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines)
and [OS GPU support](https://docs.cloud.google.com/compute/docs/images/os-details).

## Install the driver and run

After SSHing into the VM, install the driver using Google's official installer:

```bash
curl -fSsL https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz -o /tmp/cuda_installer.pyz
sudo python3 /tmp/cuda_installer.pyz install_driver --installation-mode=repo --installation-branch=prod
nvidia-smi
```

Reconnect if the driver installation requires a reboot, and confirm `nvidia-smi`
lists the L4 before training. [Official driver instructions](https://docs.cloud.google.com/compute/docs/gpus/install-drivers-gpu).
The launcher installs the pinned CUDA PyTorch wheel; a separate CUDA toolkit is
not needed for these wheel-based commands.

Clone the private repository's `r13` branch using your existing GitHub access,
and place the organizer's `student_resource` on the Persistent Disk. Its
`dataset` and `utils/validate_submission.py` must both be present. Then run:

```bash
bash scripts/vm_r13.sh /absolute/path/to/student_resource
tail -f work/r13/runner.log
```

The launcher uses native BF16 and fused AdamW for the CE, 30 CPU workers with
enough queued normalization work, exact GPU neighbour search, 6-million-pair
feature shards, 64-row CE training batches and up-to-1,024-row inference
batches. It checks native BF16 and CUDA before the expensive work starts.
Only one training job should use this VM at a time.

After the job exits, the supervisor schedules shutdown in ten minutes, keeps
the runner's exit status and never postpones an earlier overall deadline.
`R13_SHUTDOWN_ON_EXIT=0` opts out of early shutdown, retaining the overall cap.
To continue after an interruption, start the same VM if stopped, verify its
runtime limit, and use the same command with `--resume`. Check the active
shutdown schedule when restarting; changing code/settings requires fresh work.

Download `work/r13/result.json`, `metrics.json`, logs and the validated
`output/r13/*.tsv`. Preserve models if needed. Delete the VM and retained disk
only when those results are safely saved and no further resume is needed.

Full-cloud throughput, peak RAM and the **98.5% website target remain
unmeasured**. Correctness tests cannot establish a leaderboard score.
