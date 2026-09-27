# R14 pre-fusion handoff status — 27 September 2026

The user designated **`codex/r16-fast-ce-fusion`** as the location for all
available handoff deliverables. The transfer has **not** been completed.

| Deliverable | Status / location |
| --- | --- |
| R16 model and bounded runner | Published in commit `33b2edf` on this branch |
| Model validation report | [66-test implementation review](r16_fast_ce_fusion_review.md) |
| IAP / Drive handoff for the R14 teammate | [Copy-ready handoff](../docs/r14_iap_transfer.txt) |
| Manifest schema | [Example](../docs/r16_prefusion_inputs.example.json); placeholders remain intentional |
| Original graph, CE-A, swapped CE-A, CE-B score files | Not supplied to this session; actual raw paths and schemas remain unverified |
| Original R14 ID mappings | Reported at `D:\star_r10b\work\norm`; not supplied here |
| Completed production `inputs.json` | Pending inspection of actual raw files and producing-run provenance |
| Transfer archive and checksums | Not produced; source files are unavailable here |
| Upload to `~/r14_prefusion/` on the VM | Not performed |
| VM preflight | Not run |

The destination is `amazon-r9-train`, project `amazon-ml-r9`, zone
`asia-south1-c`, SSH user `akshayvijaygudur_gmail_com`. It has **no external IP**;
use authenticated Google Cloud IAP. The user reports that the public key
commented `r12fr-pull` is installed in the VM's `authorized_keys`; the holder
must use the matching existing private key locally. No private key is requested
in chat or included in repository deliverables.

The raw files are reported to be on the R14 teammate's Windows machine.
This session is on the Mac workspace, with no supplied raw R14 archive,
shared Drive link or authenticated gcloud connection. AnyDesk access was not
approved, so that remote machine was not accessed. A GitHub branch is a
handoff location, not access to those source files or the VM.

The teammate should inspect the existing raw component files, complete the
manifest with actual relative paths and score schemas, and either upload the
folder through IAP or share the prepared archive and SHA-256 through Drive.
Record the source inventory, checksum/transfer verification, destination
inventory and preflight result here once they exist. Never mark the example
manifest complete by substituting the three final `fusion_ab` predictions.

No new R12 export, neural training, submission, IAM change or shutdown change
has been performed. The original R12 fallback and gate remain unchanged. The
reported shutdown is **2026-09-27 17:44:11 UTC**, with the R16 runner's latest
work deadline **17:34:11 UTC** for that session.
