import argparse


def main():
    parser = argparse.ArgumentParser(description="Reproducible CPU entity-resolution baseline")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Build a sampled training/development dataset")
    prep.add_argument("--dataset", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--anchors", type=int, default=6000)
    prep.add_argument("--background", type=int, default=200000)
    prep.add_argument("--seed", type=int, default=42)
    idx = sub.add_parser("index", help="Build a country-aware disk search index")
    idx.add_argument("--sources", nargs="+", required=True)
    idx.add_argument("--output", required=True)
    train = sub.add_parser("train", help="Retrieve candidates, train, tune and evaluate")
    train.add_argument("--dev", required=True)
    train.add_argument("--index", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--per-channel", type=int, default=20)
    train.add_argument("--iterations", type=int, default=450)
    train.add_argument("--threads", type=int, default=6)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--posting-budget", type=int, default=0, help="Per-channel term posting budget; zero preserves unrestricted retrieval")
    pred = sub.add_parser("predict", help="Write matching_results.tsv and candidate_pairs.tsv")
    pred.add_argument("--anchors", required=True)
    pred.add_argument("--index", required=True)
    pred.add_argument("--model", required=True)
    pred.add_argument("--output", required=True)
    parallel = sub.add_parser("predict-parallel", help="Checkpointed parallel inference")
    parallel.add_argument("--anchors", required=True)
    parallel.add_argument("--index", required=True)
    parallel.add_argument("--model", required=True)
    parallel.add_argument("--output", required=True)
    parallel.add_argument("--workers", type=int, default=4)
    parallel.add_argument("--chunk-size", type=int, default=128)
    parallel.add_argument("--resume", action="store_true")
    parallel.add_argument("--max-chunks", type=int, help="Process at most this many new chunks; useful for resume testing")
    validate = sub.add_parser("validate", help="Stream full submission checks, including target ID existence")
    validate.add_argument("--anchors", required=True)
    validate.add_argument("--targets", nargs="+", required=True)
    validate.add_argument("--output", required=True)
    validate.add_argument("--receipt", required=True)
    report = sub.add_parser("report", help="Summarize a saved development run")
    report.add_argument("--run", required=True)
    report.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.anchors < 100 or args.background < 0:
            parser.error("Use at least 100 anchors and a nonnegative background count")
        from .prepare import prepare
        prepare(args.dataset, args.output, args.anchors, args.background, args.seed)
    elif args.command == "index":
        from .retrieval import build_index
        build_index(args.sources, args.output)
    elif args.command == "train":
        if min(args.per_channel, args.iterations, args.threads) < 1:
            parser.error("Channel size, iterations and thread count must be positive")
        if args.posting_budget < 0:
            parser.error("Posting budget must be nonnegative")
        from .model import train
        train(args.dev, args.index, args.output, args.per_channel, args.iterations, args.threads, args.seed, args.posting_budget)
    elif args.command == "predict":
        from .model import predict
        predict(args.anchors, args.index, args.model, args.output)
    elif args.command == "predict-parallel":
        from .parallel import predict_parallel
        predict_parallel(args.anchors, args.index, args.model, args.output, args.workers, args.chunk_size, args.resume, args.max_chunks)
    elif args.command == "validate":
        from .validate import validate_submission
        validate_submission(args.anchors, args.targets, args.output, args.receipt)
    elif args.command == "report":
        from .report import summarize
        summarize(args.run, args.output)


if __name__ == "__main__":
    main()
