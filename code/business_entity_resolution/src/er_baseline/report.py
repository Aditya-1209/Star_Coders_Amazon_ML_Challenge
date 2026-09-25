from collections import Counter
import json
from pathlib import Path


def summarize(run, output):
    run, output = Path(run), Path(output)
    result = json.loads((run / "metrics.json").read_text())
    errors = json.loads((run / "holdout_errors.json").read_text())
    manifest = result["development_manifest"]
    indexed_targets = result["index"]["records"]
    full_corpus = indexed_targets == manifest["full_training_target_pool"]
    categories = Counter((e["error"], e["retrieved"]) for e in errors)
    holdout = result["partitions"]["holdout"]
    text = [
        "# CPU baseline: " + ("full training candidate corpus" if full_corpus else "reduced candidate corpus"), "",
        "This is a development experiment using sampled Source 1 businesses. It is not a leaderboard score or a full-test submission.", "",
        f"The model uses {manifest['anchors']:,} sampled Source 1 businesses and searches {indexed_targets:,} Source 2/3 records. " + ("The index contains the entire training target pool." if full_corpus else f"The reduced index contains {manifest['labeled_positive_targets']:,} labeled matches plus {manifest['background_targets']:,} random distractors; the full training target pool contains {manifest['full_training_target_pool']:,} records."), "",
        "## Results", "",
        "| Partition | Source 1 businesses | Macro F0.5 | Pair precision | Pair recall | Candidate recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in result["partitions"].items():
        text.append(f"| {name} | {metrics['anchors']:,} | {metrics['macro_f05']:.4f} | {metrics['pair_precision']:.2%} | {metrics['pair_recall']:.2%} | {metrics['candidate_pair_recall']:.2%} |")
    text += [
        "", f"The threshold **{result['config']['threshold']:.3f}** was selected on the tuning partition. The holdout businesses were not used to fit the classifier or choose its threshold.", "",
        f"- Holdout candidate oracle macro F0.5: **{holdout['candidate_oracle_macro_f05']:.4f}**. This is the score a perfect classifier could achieve using only the retrieved candidates.",
        f"- Always predicting no matches would score **{holdout['empty_prediction_baseline_f05']:.4f}** on this holdout.",
        f"- Holdout singleton accuracy: **{holdout['singleton_accuracy']:.2%}**, across {holdout['singletons']} businesses with no matches.",
        f"- Mean candidates per holdout business: **{holdout['mean_candidates']:.1f}**.", "",
        "| Holdout country | Businesses | Macro F0.5 | Pair precision | Pair recall |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for country, metrics in holdout["countries"].items():
        text.append(f"| {country} | {metrics['anchors']:,} | {metrics['macro_f05']:.4f} | {metrics['pair_precision']:.2%} | {metrics['pair_recall']:.2%} |")
    text += [
        "", "## Error breakdown", "",
        f"- False positive pairs: {categories[('false_positive', True)]:,}.",
        f"- True pairs missed by retrieval: {categories[('false_negative', False)]:,}.",
        f"- Retrieved true pairs rejected by the classifier: {categories[('false_negative', True)]:,}.",
        "", "## CPU and memory", "",
        "Measured on an Apple M4 with 16 GB RAM; CatBoost used six CPU threads and no GPU.", "",
        f"- Development sample preparation: {manifest['elapsed_seconds']:.1f} seconds.",
        f"- Search index construction: {result['index']['build_seconds']:.1f} seconds.",
        f"- Candidate retrieval and feature generation for all {manifest['anchors']:,} businesses: {result['retrieval_and_features_seconds']:.1f} seconds.",
        f"- Classifier fitting: {result['training_seconds']:.1f} seconds.",
        f"- Training command total through scoring: {result['total_seconds']:.1f} seconds.",
        f"- Peak training-process resident memory: {result['peak_process_memory_mb']:.1f} MiB (not whole-system memory).",
        f"- Candidate pairs scored: {result['candidate_pairs']:,}.",
        "", "## Most influential features", "",
    ]
    text.extend(f"- `{name}`: {importance:.2f}" for name, importance in result["feature_importance"][:8])
    text += [
        "", "## Interpretation and next step", "",
        ("The full training target corpus has been evaluated. Review retrieval recall, error rates, and query throughput before full test inference. Country-level results do not establish performance on unseen France."
         if full_corpus else "The CPU pipeline is operational. The next experiment should increase the candidate corpus toward the full training pool, then reevaluate retrieval recall, false positives, speed, and threshold choice. Do this before full test inference; extra distractors can change both rankings and accuracy."),
    ]
    if full_corpus:
        serial_hours = result["retrieval_and_features_seconds"] / manifest["anchors"] * 1732544 / 3600
        text += ["", f"Full test inference has not been started. At the measured serial retrieval/feature speed, processing all 1,732,544 test anchors would take roughly **{serial_hours:.1f} hours**, before additional overhead and changes in country distribution. Parallel inference needs its own benchmark."]
    text += ["", "## Limits", ""]
    text.extend(f"- {limitation}" for limitation in result["limitations"])
    text += ["- Holdout exports are development predictions only. No full-test predictions have been generated or uploaded."]
    validation_path = run / "validation.json"
    if validation_path.exists():
        validation = json.loads(validation_path.read_text())
        text += [
            "", "## Verification", "",
            f"- {validation['unit_tests_passed']} unit tests passed, covering scoring, grouping, Unicode, missing fields, and search ranking.",
            f"- Official validator with ID-existence checks: {validation['official_validator_with_check_ids']}.",
            f"- Strict candidate subset and valid-ID audit: {validation['strict_candidate_subset_and_valid_id_audit']}.",
            f"- Independent score calculation from exported TSV files: {validation['independent_exported_tsv_metric_recalculation']}.",
            f"- Reloaded model reproduced both exports exactly on {validation['saved_model_inference_exact_match_anchors']} smoke-test businesses, including singletons.",
        ]
    text += ["", "## Reproduction", "", f"See `code/business_entity_resolution/README.md` for commands, pinned dependencies, and model details. Machine-readable metrics, model weights, pair features, predictions, and errors are in `{run}`.", ""]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(text), encoding="utf-8")
    print(f"Wrote {output}")
