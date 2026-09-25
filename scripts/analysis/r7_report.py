"""Write an honest local result summary after validation, with output hashes."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    stage2 = read(args.work / "models/metrics.json")
    stage3 = read(args.work / "models/stage3_metrics.json")
    outputs = {}
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        path = args.output / name
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        outputs[name] = {"bytes": path.stat().st_size, "sha256": digest}
    report = {"metric": "macro F0.5, including singletons and unretrieved positives",
              "selection_fold": 3, "holdout_fold": 4,
              "amazon_score": None, "amazon_score_status": "requires submitting to Amazon; not measured locally",
              "stage2_holdout": stage2["fold4_stage2_excl"],
              "stage3_holdout": stage3["fold4_selected"],
              "stage3_by_country": stage3["fold4_by_country"],
              "stage3_oracle": stage3["fold4_oracle"],
              "stage2_feature_trials": stage2.get("feature_trials", []),
              "stage3_feature_trials": stage3.get("feature_trials", []),
              "shift_check": read(args.work / "shift_check.json"),
              "retrieval_audit": read(args.work / "retrieval_audit.json"),
              "outputs": outputs}
    (args.work / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    s2, s3 = report["stage2_holdout"], report["stage3_holdout"]
    text = ("# r7 overnight experiment\n\n"
            "These are local held-out macro F0.5 scores, not Amazon leaderboard accuracy. "
            "All configuration, feature and blend selection used fold 3; fold 4 was used only for reporting.\n\n"
            f"| Model | Macro F0.5 | Precision | Recall |\n|---|---:|---:|---:|\n"
            f"| Stage 2 | {s2['macro_f05']:.6f} | {s2['pair_precision']:.6f} | {s2['pair_recall']:.6f} |\n"
            f"| Selected stage 2/3 blend | {s3['macro_f05']:.6f} | {s3['pair_precision']:.6f} | {s3['pair_recall']:.6f} |\n\n"
            "The baseline feature ablation uses the same expanded candidates and stage-1 models. "
            "It is not a rerun of the old Amazon submission. France has no labeled holdout; "
            "reaching 97–98 on Amazon remains unverified.\n\n"
            "Detailed results, ablations, diagnostics and output hashes are in `result.json` alongside this report.\n")
    (args.work / "result.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
