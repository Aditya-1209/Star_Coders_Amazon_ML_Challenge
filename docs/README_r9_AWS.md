# R9: accuracy experiments on one AWS GPU

**Status: implementation prepared; R9 has not been trained, benchmarked or deployed.**
Runtime tests are included and run as the first pipeline stage on AWS. Only source
and configuration checks were performed while preparing this branch. No R9 score
or AWS runtime is available yet.

R7 reached **0.967381 local macro F0.5**, with **0.995076 pair precision** and
**0.930945 pair recall**. These are different metrics; none is a new Amazon score.
See [the measured report](../reports/r7_overnight_result.md). The target of 97–98
on Amazon is an experiment objective, not a promised result.

## Why try this alongside R8?

R8 explores neural networks. R9 explores **retrieval and the macro objective**:

1. Rebuild the R7 pipeline as a reference in its own cache. Learn supervised
   transliteration using folds 0/1/8/9 only, keeping graph-training folds out of
   that dictionary. This means the rebuilt reference can differ from historical R7.
2. Train two provisional graph matchers on folds 6 and 7 separately. Each scores
   the other fold; both are averaged for validation and test businesses. A
   model's predictions on its own training businesses never become its graph inputs.
3. Use confident graph matches (probability ≥ 0.8) as new queries, with up to 15
   neighboring targets per anchor. Expand once more, retain all old candidates,
   and recompute record-to-record support. This can reach matches the first
   retrieval round missed. Extra hops can also spread false matches: measure both.
4. Fit two final XGBoost models: ordinary pair loss and equal total loss weight
   per business. Compare each and their mean, with three fixed baseline blend
   weights. Equal-business weighting is a surrogate for macro F0.5, not an exact
   optimization of the metric. This is a small, predefined comparison, not a huge sweep.
5. Split tuning fold 3 into deterministic halves. Select stopping iterations,
   model/blend and cutoff on 3A. Gate that **one** proposal on 3B. Require at least
   +0.0005 macro F0.5, a positive approximate paired lower bound, and no seen-country
   drop worse than 0.002. Otherwise keep the rebuilt baseline, skipping the extra
   test expansion. Fold 4 is evaluated after selection is frozen.

3B is a guard for the new final layer, not an independent holdout: earlier baseline
models already used fold 3. Shared targets also make the paired normal bound
approximate. Fold 4 has been seen in previous R7 work; this branch never uses its
score to select a configuration. A truly new blind assessment still requires
unseen labels or Amazon evaluation.

France accounts for roughly 15% of test Source 1 businesses and has no training
labels. Improving US/India validation alone may fail to move the leaderboard.
The optional country-transfer diagnostic below excludes an entire country from
**supervised normalization, every model fit, stopping and selection**. It measures
transfer to that held-out country, not France accuracy. No pseudo-labels are treated
as ground truth. No external models, data or APIs are used for matching.

## Machine, scale and budget

Start with **one `g5.4xlarge` in `us-east-1`**: 16 vCPUs, 64 GiB host RAM and one
NVIDIA A10G with 24 GB advertised GPU memory. The extra host RAM matters because
record tables, graph joins and quantized matrices still use RAM. The template also
allows `g6.4xlarge` and `g4dn.4xlarge` if price/capacity is better; the latter has
less GPU memory. [AWS G5 specifications](https://aws.amazon.com/ec2/instance-types/g5/)

- One heavy stage and one GPU model at a time; 12 CPU workers by default.
- Country-at-a-time retrieval and pair features; batched predictions and matrices.
- One additional graph round with a fixed neighbor cap. This is not all-pairs matching.
- Cache preprocessing and graph features on **200 GiB encrypted gp3 EBS**. Reuse
  completed stages when resuming the same run, with code/data/settings checks.
- No multi-GPU or multi-node training. GPU time will not accelerate string matching
  and graph joins proportionally. Measure stage runtimes before buying more GPUs.
- Start On-Demand. Spot interruption recovery is not validated for this release.

The friend’s laptop only uploads files and controls EC2. The experiment is a
systemd service on AWS; closing that laptop or disconnecting the browser does
not stop it.

**Price is checked at launch, not hardcoded.** Run `estimate_cost.py` below for
the current Linux On-Demand rate in your region. With $100 total credit, reserve
at least $20 for storage, networking and margin. For example, **if** the returned
rate is $1.624/hour, $80 buys about 49.3 compute hours, and a 12-hour session is
about $19.49 in compute. This arithmetic is illustrative, not a current price quote
or a claim that R9 finishes in 12 hours. [AWS Price List API](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/using-price-list-query-api.html)

At a gp3 rate of $0.08/GB-month, 200 GB costs $16 for a full month retained, billed
proportionally. Check your region; S3, public IPv4 and transfer are additional.
Stopping EC2 stops compute billing but **EBS and S3 remain billable**. The timer is
a wall-time guard, not a dollar cap; allow shutdown/upload grace and other account
usage. [EBS pricing](https://aws.amazon.com/ebs/pricing/),
[EC2 stop/start billing](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Stop_Start.html)

## 1. Account checks before launching

In the friend's AWS account:

1. Confirm remaining credits, expiry and EC2 eligibility in Billing. Free-plan
   account restrictions can prevent GPU use even when credits are visible; check
   the account plan before proceeding. This repository never changes billing plans.
   [Credit terms](https://aws.amazon.com/awscredits/),
   [EC2 Free Tier eligibility](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-free-tier-usage.html)
2. In **Service Quotas → EC2**, check **Running On-Demand G and VT instances** in
   `us-east-1`. Request **16 vCPUs** if necessary. The default can be zero.
   [AWS quotas](https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html)
3. Set a Billing budget alert for your experiment allowance. Alerts do not stop
   spend. Check for any other running resources sharing the credit.
4. Choose an existing VPC and a **public subnet** in it, with an Internet Gateway
   route. The template opens no inbound ports and uses Session Manager. A private
   subnet without outbound access will fail setup. Do not add a NAT gateway just
   for this experiment.
5. Create a private S3 bucket in `us-east-1`, with Block Public Access enabled and
   default SSE-S3 encryption. A custom KMS key requires additional IAM permissions
   not included here. Keep the organizer dataset private.

## 2. Prepare the source and dataset on the friend's laptop

Use Git Bash, Linux, macOS or WSL. The repository is private; authenticate with a
GitHub account that has access. No GitHub credentials go on the AWS machine.

```bash
git clone --branch r9 --single-branch https://github.com/Aditya-1209/Star_Coders_Amazon_ML_Challenge.git
cd Star_Coders_Amazon_ML_Challenge
git archive --format=tar.gz --output=r9-source.tar.gz HEAD
```

If already cloned, save local edits, then `git fetch origin` and `git switch r9`
(or `git switch --track origin/r9` on first use), followed by `git pull --ff-only`.
Archive the committed revision you intend to run; uncommitted files are not included.

Using the **S3 console → bucket → Upload**, upload:

| Local file | S3 object key |
|---|---|
| `r9-source.tar.gz` | `r9/input/source.tar.gz` |
| The organizer's original dataset ZIP | `r9/input/dataset.zip` |

The ZIP must contain the six source TSVs, training truth and the official
`utils/validate_submission.py`. Extraction accepts the organizer's known path
layouts and refuses incomplete or conflicting files. No dataset is pushed to GitHub.

## 3. Check pricing and launch from AWS CloudShell

Open **CloudShell in us-east-1**. Fill in your actual bucket, VPC and subnet values:

```bash
export AWS_DEFAULT_REGION=us-east-1
export AWS_PAGER=''
R9_BUCKET=your-private-bucket-name
R9_VPC=vpc-your-id
R9_SUBNET=subnet-your-public-subnet-id
R9_RUN=first-run

aws s3 cp "s3://$R9_BUCKET/r9/input/source.tar.gz" /tmp/r9-source.tar.gz
mkdir -p ~/r9-launch
tar -xzf /tmp/r9-source.tar.gz -C ~/r9-launch
cd ~/r9-launch
python3 scripts/aws/estimate_cost.py --instance-type g5.4xlarge --region us-east-1 --credit 100 --reserve 20 --hours 12
aws cloudformation validate-template --template-body file://infra/r9-ec2.yaml
R9_SHA=$(sha256sum /tmp/r9-source.tar.gz | cut -d ' ' -f 1)
```

The pricing helper only queries AWS; it does not launch anything. If pricing access
is denied, ask the account owner for `pricing:GetProducts` or verify the rate in
the AWS calculator/console. The deployer also needs CloudFormation, EC2, IAM role
creation/pass-role, SSM parameter-read and S3 permissions. Instance permissions
are scoped to the two input objects, its result prefix and Session Manager.

**The next command creates paid resources and starts the experiment. Run it only
when the account, quota and displayed cost are ready.** This command has not been
executed while preparing the branch.

```bash
aws cloudformation deploy \
  --template-file infra/r9-ec2.yaml \
  --stack-name "r9-$R9_RUN" \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    VpcId="$R9_VPC" SubnetId="$R9_SUBNET" Bucket="$R9_BUCKET" \
    RunId="$R9_RUN" SourceSha256="$R9_SHA" \
    InstanceType=g5.4xlarge WallHours=12 VolumeGiB=200

aws cloudformation describe-stacks --stack-name "r9-$R9_RUN" --query 'Stacks[0].Outputs'
```

The template resolves the AWS-owned Ubuntu 24.04 NVIDIA DLAMI through its
[official public SSM parameter](https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-x86-base-gpu-ami-ubuntu-24-04.html).
It checks the source archive hash, installs pinned Python dependencies, extracts
data, checks that XGBoost really uses CUDA, runs synthetic tests, then starts the
full pipeline. Stack `CREATE_COMPLETE` means the instance exists; it does **not**
mean setup/tests/training succeeded.

The boot timer is armed before downloads/installations. With `WallHours=12`, the
runner gets at most 11 hours per invocation and the boot guard requests shutdown
at 12 hours, plus shutdown grace. Completion, stage failure or the runner time
limit triggers a bounded result upload and instance stop. A failed upload leaves
the files on the stopped EBS disk. The service is intentionally **not enabled on
boot**, so restarting the instance cannot silently start another training session.

## 4. Inspect progress and recover

Use **EC2 → instance → Connect → Session Manager**. Allow a few minutes for SSM
registration, then:

```bash
sudo tail -n 60 /var/log/r9-bootstrap.log
sudo systemctl status r9.service --no-pager
sudo cat /opt/r9/repo/work/r9/run.json
sudo tail -n 50 /opt/r9/repo/work/service.log
```

`run.json` gives the active stage and its log path. For example:

```bash
sudo tail -f /opt/r9/repo/work/r9/logs/expand_train.log
```

A pipeline failure stops compute after exporting available diagnostics. Inspect
`bootstrap.log`, `service.log` and `work/logs/` in the S3 result prefix first.
The included runtime tests have not been executed during source preparation;
if they fail, stop and fix that failure before the real run. Do not skip the gate.

After a wall-time interruption, recheck your credit, start the **same stopped
instance** in EC2, connect with Session Manager and explicitly resume:

```bash
sudo systemctl start --no-block r9.service
```

This automatically supplies `--resume` if `work/r9/run.json` exists. Completed,
compatible stages are skipped; an interrupted stage restarts from its beginning.
There is no checkpoint inside an individual XGBoost fit or graph-building stage.
The boot timer is armed again. Do not start a duplicate instance to resume, change
source/dependencies mid-run or reuse R7 caches: the normalization protocol changed.

If the instance stops before creating `run.json`, inspect bootstrap/service logs;
a setup failure may require completing setup manually or replacing the stack.
Keep the first instance stopped while investigating. Do not run two GPU instances
against the same S3 `RunId`.

## 5. Results, submission and cleanup

On completion, look under `s3://YOUR-BUCKET/r9/results/YOUR-RUN/`:

| Path | Meaning |
|---|---|
| `work/result.md`, `work/result.json` | Aggregate measured result, settings, timings and hashes |
| `work/metrics.json` | Fold-4 metrics and country breakdown after selection froze |
| `work/selection.json` | 3A trials, 3B acceptance gate, selected baseline or R9 |
| `work/expand_train.json` | Candidate growth from the extra graph round |
| `work/base/shift_check.json` | Train/test separability diagnostic, not accuracy |
| `work/logs/validate.log` | Official full formatting/ID checks |
| `output/matching_results.tsv` | Selected final matches |
| `output/candidate_pairs.tsv` | Candidate list corresponding to those matches |
| `output/baseline/` | Rebuilt reference output for comparison |

Submission files are ready only if `result.json` says `status: complete` and
`official_validation: PASS`. The official validator checks format/IDs, not accuracy.
The report also scores the R9 proposal on fold 4 for diagnosis even when the 3B
gate rejected it. That score never changes the frozen selection.
Download both selected TSVs for the organizer's submission workflow; this branch
does not submit to Amazon automatically. Review the actual Amazon score separately.
Models and logs are uploaded; large features and normalized data remain only on EBS.

Once the results are safely downloaded and you no longer need resume caches,
delete the `r9-YOUR-RUN` CloudFormation stack. **That terminates its instance and
deletes its EBS disk.** The pre-existing S3 bucket remains: remove unwanted inputs
and artifacts manually after saving what you need. Confirm there are no leftover
instances/volumes and check Billing. Stopped instances still retain storage charges.

## Optional: unseen-country diagnostic within the remaining budget

Do this after the first result, not concurrently. Launch a separate stack with a
new `RunId` and add `ExcludeCountry=India`. It rebuilds its own normalization and
features, trains on US supervision only, selects on US fold 3, and reports India
fold 4 without tuning against it. It stops after evaluation, without full test
inference or submission generation. `ExcludeCountry=US` reverses the diagnostic.

This costs another preprocessing/training run. Check measured time and remaining
credit before deciding. Never reuse a full-supervision transliteration map or
models for this diagnostic. Even a good India transfer result is not a France score.

## Direct Linux invocation / later R8 ensemble

For an already provisioned Linux machine with Python 3.12, NVIDIA drivers and
the pinned dependencies, the equivalent commands are below. These commands **run
the experiment**; they are not needed on the controlling laptop.

```bash
.venv/bin/python scripts/prepare_r9_data.py /path/to/organizer-dataset.zip
.venv/bin/python scripts/run_r9.py --plan
.venv/bin/python scripts/run_r9.py --device cuda --threads 12 --max-hours 11
# After interruption, with identical source/data/settings:
.venv/bin/python scripts/run_r9.py --device cuda --threads 12 --max-hours 11 --resume
```

The direct runner stops its child processes at its deadline; it does not stop
the host machine. The EC2 service/template supplies instance shutdown.

If R8 and R9 make different errors, an ensemble may help next. Save R8 scores by
`(sidx,tidx)` with the same source ordering and fold definition. Blend **out-of-fold
probabilities**, keeping candidate unions and exclusive target ownership; never
blend final match lists or fit a combiner using in-sample neural predictions.
R9 preserves scored test predictions when selected, and baseline scores in its
cache. Automated R8 integration is not implemented because its score format and
validation protocol have not been provided.
