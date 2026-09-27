# Google Cloud setup for R13

R13 now targets **your existing Mumbai `g2-standard-32` VM**: one L4,
128 GB RAM and a **200 GB balanced Persistent Disk**. Keep this machine and
disk for the initial run; the software does not require the earlier guide's
300 GB SSD configuration. Full-data disk usage and runtime remain unmeasured.

| Console setting | Choose |
| --- | --- |
| Machine family | GPU / G2 |
| Machine type | `g2-standard-32` |
| GPU | 1 NVIDIA L4, 24 GB VRAM, included with G2 |
| vCPUs / RAM | 32 / 128 GB |
| Region / zone | Mumbai `asia-south1` / `asia-south1-c` |
| Provisioning model | Standard (on-demand) |
| OS | Ubuntu 24.04 LTS, x86-64, standard image |
| Python | 3.12 (your installed 3.12.3 is suitable) |
| Boot/work disk | Your existing 200 GB balanced Persistent Disk (`pd-balanced`) |
| Maximum VM runtime / action | 12 hours / STOP |
| Automatic restart | Disabled |

The zone and current disk configuration come from your VM screenshot. G2 machine
specifications are documented in [Google's machine documentation](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines).
The maximum runtime is a recommended setting, not something established by the
screenshot. Confirm it in Console before starting a new run.

`g2-standard-24` has **two** L4s, and R13 would leave one unused. The smaller
`g2-standard-16` has the same single GPU but half the host memory and CPU count;
the 128 GB profile is intended for the larger machine. An A100 is an option
after profiling shows GPU time dominates, but its higher hourly cost alone
does not establish a faster or cheaper end-to-end run.

## Balanced-disk and RAM tuning

The VM profile allows **12 GiB of RAM for the active split's CE token cache**.
When it fits, tokens and lengths are copied once into read-only NumPy arrays,
avoiding random disk reads during shuffled training and length-sorted scoring.
An oversized cache or a Python allocation failure retains the disk-backed
memory maps. Token values, examples and model settings are unchanged. The log
reports the chosen mode and size; training metadata records both as well.

The 12 GiB budget is host RAM, not VRAM, and applies to one active CE process.
It leaves most of the 128 GB host for the model, data frames and the operating
system. On other stages, disk-backed embeddings and compressed Parquet files
remain in use. GPU stages run sequentially, and no RAM disk is required.

New package installs disable pip's download cache to avoid retaining another
copy of the CUDA wheels. The launcher preserves existing experiment caches.
The runner checks its **20 GiB free-space reserve before training and throughout
each stage**. This is a stop threshold, not a claim that all outputs fit in
180 GB. If it stops for space, retain the work directory, expand the disk and
filesystem, then resume with the same code and settings.

For context, Google's formula gives a 200 GiB zonal balanced disk a disk-side
ceiling of about **4,200 IOPS and 196 MiB/s**, before VM limits. These are maximum
limits, not measured throughput. Avoiding repeated random reads is therefore
useful to test on this disk. [Persistent Disk performance](https://docs.cloud.google.com/compute/docs/disks/performance).

Keep the dataset, work directory and model cache on the VM's persistent disk.
Reuse the existing organizer resource path; there is no need to copy the dataset
into each experiment clone. If you use Cloud Storage for transfer, keep it in
`asia-south1` and copy inputs locally before training.

## Cost on your existing VM

Your screenshot reports an earlier estimate of **about US$1.81/hour with a
10 GB disk**. At that quoted rate, 11 hours is $19.91 and 12 hours is $21.72;
these are arithmetic examples, not verified current totals. The additional
190 GB of disk, NAT/network charges and tax must be added. Use the estimate
for the existing VM in Console, with Mumbai selected, for the current total.
[Google's pricing page](https://cloud.google.com/products/compute/pricing/accelerator-optimized)
also provides region-specific rates. Retained disks and provisioned networking
can continue billing after the VM stops.

Use on-demand for the first complete experiment. The runner resumes completed
stages but restarts the interrupted active stage; a Spot eviction during
encoder/CE training can discard hours from that stage.

## Optional: recreate the same VM

Your existing VM does not need to be recreated. If creating a separate machine,
replace `YOUR_PROJECT` in Cloud Shell. **This command creates a paid VM**; it
does not upload data or start training.

```bash
gcloud compute instances create r13-l4 \
  --project=YOUR_PROJECT --zone=asia-south1-c \
  --machine-type=g2-standard-32 \
  --image-project=ubuntu-os-cloud --image-family=ubuntu-2404-lts-amd64 \
  --boot-disk-type=pd-balanced --boot-disk-size=200GB \
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

After SSHing into your existing VM, run `nvidia-smi`. If it already lists the L4,
keep the working driver. For a fresh image without a working driver, use Google's
official installer:

```bash
curl -fSsL https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz -o /tmp/cuda_installer.pyz
sudo python3 /tmp/cuda_installer.pyz install_driver --installation-mode=repo --installation-branch=prod
nvidia-smi
```

Reconnect if the driver installation requires a reboot, and confirm `nvidia-smi`
lists the L4 before training. [Official driver instructions](https://docs.cloud.google.com/compute/docs/gpus/install-drivers-gpu).
The launcher installs the pinned CUDA PyTorch wheel; a separate CUDA toolkit is
not needed for these wheel-based commands.

Use a separate clone so the running R9 checkout remains stable. Wait for R9 to
finish before starting R13 on the same GPU, and restart the VM if its previous
shutdown has stopped it. Using your existing GitHub access:

```bash
git clone --branch r13 --single-branch https://github.com/Aditya-1209/Star_Coders_Amazon_ML_Challenge.git Star_Coders_r13
cd Star_Coders_r13
```

Supply the existing organizer `student_resource` path on Persistent Disk. Its
`dataset` and `utils/validate_submission.py` must both be present. Then run:

```bash
bash scripts/vm_r13.sh /absolute/path/to/student_resource
tail -f work/r13/runner.log
```

The launcher uses native BF16 and fused AdamW for the CE, 30 CPU workers with
enough queued normalization work, exact GPU neighbour search, 6-million-pair
feature shards, 64-row CE training batches and up-to-1,024-row inference
batches. It uses 30 logical CPU workers on the 32-vCPU VM; this is a starting
configuration, not a measured claim that 30 beats the previous R9 setting of 24.
It checks native BF16, CUDA and free disk before the expensive work starts.
Only one training job should use this VM at a time.

Set `R13_CE_TOKEN_CACHE_GB=0` before a fresh run to keep token memory maps, or set
a smaller positive cap. `R13_THREADS=24` is available for CPU throughput
comparisons. Preserve these settings when resuming; changes require fresh work.

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
