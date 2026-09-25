# Running r5 on AWS

The $579 AWS Student Rewards figure is a bundle value, not $579 of EC2 credit:
$449 is Skill Builder access, $100 is a certification voucher, and up to $30
is AWS compute credit. A separate $100 sign-up credit may appear in the account.
Check the exact code's terms and **Billing and Cost Management → Credits** after
redeeming it; do not enter or share a credit code in this repository. Check
expiry, eligible services, and amount remaining before launching EC2. The AWS
account's Paid Plan allows other promotional credits but can charge beyond the
credit balance. Sources: [Student Rewards](https://www.aboutamazon.com/news/aws/aws-student-rewards-free-cloud-ai-training),
[AWS account plans](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/free-tier-plans.html),
and [viewing credits](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/useconsolidatedbilling-credits.html).

For the full three-stage run, start with a **g6.4xlarge** (16 vCPU, 64 GiB RAM,
one NVIDIA L4 with 24 GB GPU memory) if it is available and the region's
On-Demand G/VT quota permits it. The local 32 GB desktop might run out of RAM
while building the graph stage. The full corpus has not been profiled, so 64 GiB
is an initial choice, not a verified minimum. G6.8xlarge has 128 GiB if a
measured run needs more. See [AWS G6 specifications](https://aws.amazon.com/ec2/instance-types/accelerated-computing/)
and [instance quotas](https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html).
Use an Ubuntu 24.04 **x86_64 Deep Learning Base GPU AMI** with NVIDIA drivers,
a gp3 volume of at least 150 GB, and a region-specific on-demand price estimate
from the EC2 launch console or AWS Pricing Calculator. Do not assume the GPU
instance is covered by the Free Tier. Set an AWS Budget alert based on the
credit actually available. [AWS Budgets](https://docs.aws.amazon.com/cost-management/latest/userguide/budget-templates.html)
and [Deep Learning AMIs](https://docs.aws.amazon.com/dlami/latest/devguide/)
document these choices.

After transferring the repository and the **provided** `student_resource`
directory to the instance (for example, with `scp` or an S3 bucket you own),
run from the repository root:

```bash
git switch r5-recall-speed
python3.12 -m venv .venv-r5
.venv-r5/bin/python -m pip install -r code/business_entity_resolution/requirements_v2.txt
nvidia-smi
.venv-r5/bin/python scripts/run_r5.py \
  --dataset /path/to/student_resource/dataset \
  --work work/r5-full --output output/r5-full \
  --device cuda --threads 16
```

Keep `student_resource/utils/validate_submission.py` next to `dataset/`: the
runner calls the official validator after prediction. Its checkpoints and
timings are under `work/r5-full/checkpoints/`; `--resume` reuses completed
stages only when the code and inputs are identical. For an early training-only
measurement, add `--phase train`, then run the same command with `--phase
predict --resume` to finish. Inspect `work/r5-full/models/metrics.json` and
`stage3_metrics.json`: the fold-4 selected-model macro F0.5 and candidate
oracle tell us whether the 98% target is supported by held-out evidence.
Submit only the full-run `output/r5-full/matching_results.tsv` after validation.

Stop the EC2 instance when the run finishes. Compute charges stop, but EBS
storage remains billable while the volume exists; download the outputs before
deleting it. [AWS stop/start billing](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/how-ec2-instance-stop-start-works.html).
