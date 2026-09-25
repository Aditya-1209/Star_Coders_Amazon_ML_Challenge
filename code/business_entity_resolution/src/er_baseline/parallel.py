"""Bounded, resumable multiprocessing inference with atomic chunk outputs."""

import atexit
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import resource
import shutil
import sqlite3
import sys
import time

from .data import read_records

SCHEMA = 1
FILES = ("matching_results.tsv", "candidate_pairs.tsv")
HEADERS = ("source1_entity_id\tmatched_entity_ids\n", "source1_entity_id\tcandidate_entity_ids\n")
_WORKER = None


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def fingerprint(path, content_hash=False):
    path = Path(path).resolve()
    stat = path.stat()
    value = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if content_hash:
        value["sha256"] = digest(path)
    return value


@contextmanager
def output_lock(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".inference.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another inference process owns {output}") from error
        yield


def _worker_init(index, model_dir, parts):
    global _WORKER
    from catboost import CatBoostClassifier
    from .retrieval import Retriever
    from .text import FEATURE_NAMES
    config = json.loads((Path(model_dir) / "config.json").read_text())
    if config["feature_names"] != FEATURE_NAMES:
        raise ValueError("Model feature schema differs from inference code")
    model = CatBoostClassifier()
    model.load_model(str(Path(model_dir) / "model.cbm"))
    retriever = Retriever(index, config["per_channel"], config.get("posting_budget", 0))
    # Read-only workers can share file-backed pages through the OS instead of
    # copying every index read into each process's SQLite cache.
    retriever.db.execute("PRAGMA mmap_size=4294967296")
    _WORKER = (model, retriever, config, Path(parts))
    atexit.register(retriever.close)


def _chunk_digest(records):
    return hashlib.sha256(json.dumps([r.values() for r in records], ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _part_paths(parts, chunk_id):
    stem = f"{chunk_id:06d}"
    return [parts / f"{stem}.{name}" for name in FILES], parts / f"{stem}.json"


def _worker_chunk(chunk_id, records, input_digest):
    import numpy as np
    from .text import features
    model, retriever, config, parts = _WORKER
    started = time.monotonic()
    candidate_ids, offsets, x = [], [0], []
    for anchor in records:
        retrieved = retriever.query(anchor)
        ids = [target.entity_id for target, _ in retrieved]
        if len(ids) != len(set(ids)) or any(not target.startswith(("S2-", "S3-")) for target in ids):
            raise ValueError("Retriever emitted duplicate or invalid target IDs")
        candidate_ids.append(ids)
        x.extend(features(anchor, target, ranks) for target, ranks in retrieved)
        offsets.append(len(x))
    retrieval_seconds = time.monotonic() - started
    prediction_started = time.monotonic()
    scores = model.predict_proba(np.asarray(x, dtype=np.float32), thread_count=1)[:, 1] if x else np.empty(0)
    prediction_seconds = time.monotonic() - prediction_started
    lines = [[], []]
    match_count = empty_matches = 0
    for i, (record, ids) in enumerate(zip(records, candidate_ids)):
        accepted = [target for target, score in zip(ids, scores[offsets[i]:offsets[i + 1]]) if score >= config["threshold"]]
        lines[0].append(record.entity_id + "\t" + ",".join(sorted(accepted)) + "\n")
        lines[1].append(record.entity_id + "\t" + ",".join(sorted(ids)) + "\n")
        match_count += len(accepted)
        empty_matches += not accepted
    paths, metadata_path = _part_paths(parts, chunk_id)
    output_fingerprints = []
    for path, content in zip(paths, lines):
        data = "".join(content).encode("utf-8")
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        output_fingerprints.append({"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    metadata = {
        "chunk_id": chunk_id, "input_digest": input_digest, "rows": len(records),
        "first_id": records[0].entity_id, "last_id": records[-1].entity_id,
        "candidate_pairs": len(x), "matched_pairs": match_count, "empty_matches": empty_matches,
        "files": output_fingerprints, "worker_pid": os.getpid(),
        "worker_peak_mib": rss / (1024 * 1024 if sys.platform == "darwin" else 1024),
        "retrieval_features_seconds": retrieval_seconds, "prediction_seconds": prediction_seconds,
        "seconds": time.monotonic() - started,
    }
    # This marker is written last: chunks without a marker are safe to redo.
    atomic_json(metadata_path, metadata)
    return metadata


def _completed_part(parts, chunk_id, records, input_digest):
    paths, metadata_path = _part_paths(parts, chunk_id)
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text())
    if metadata["chunk_id"] != chunk_id or metadata["input_digest"] != input_digest or metadata["rows"] != len(records):
        raise ValueError(f"Checkpoint input mismatch in chunk {chunk_id}")
    for path, expected in zip(paths, metadata["files"]):
        if not path.is_file() or path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError(f"Checkpoint corruption in chunk {chunk_id}: {path}")
    return metadata


def _scan_anchors(path):
    seen, countries = set(), Counter()
    for record in read_records(path):
        if not record.entity_id.startswith("S1-") or record.entity_id in seen:
            raise ValueError(f"Invalid/duplicate Source 1 ID: {record.entity_id}")
        seen.add(record.entity_id)
        countries[record.country] += 1
    return {"rows": len(seen), "countries": dict(countries)}


def _chunks(path, chunk_size):
    iterator = iter(read_records(path))
    for chunk_id in itertools.count():
        batch = list(itertools.islice(iterator, chunk_size))
        if not batch:
            return
        yield chunk_id, batch, _chunk_digest(batch)


def predict_parallel(anchors_path, index, model_dir, output, workers=4, chunk_size=128, resume=False, max_chunks=None):
    """Run bounded workers, checkpoint outputs, and merge only a complete run."""
    if workers < 1 or chunk_size < 1 or (max_chunks is not None and max_chunks < 1):
        raise ValueError("Workers, chunk size, and max chunks must be positive")
    output, index, model_dir, anchors_path = map(Path, (output, index, model_dir, anchors_path))
    started = time.monotonic()
    with output_lock(output):
        manifest_path = output / "inference_manifest.json"
        if manifest_path.exists() and not resume:
            raise FileExistsError("Inference already exists; use --resume to continue it")
        if not manifest_path.exists() and any(p.name != ".inference.lock" for p in output.iterdir()):
            raise FileExistsError("Refusing to mix inference with existing output files")
        with sqlite3.connect(f"file:{index.resolve()}?mode=ro", uri=True) as db:
            index_meta = {key: json.loads(value) for key, value in db.execute("SELECT key,value FROM metadata")}
        # File metadata detects an altered index without rehashing a multi-GB
        # database at every restart. Input text, weights and code are hashed.
        identity = {
            "schema": SCHEMA, "chunk_size": chunk_size,
            "anchors": fingerprint(anchors_path, True), "index": fingerprint(index),
            "index_metadata": index_meta,
            "model": fingerprint(model_dir / "model.cbm", True),
            "config": fingerprint(model_dir / "config.json", True),
            "code": {name: digest(Path(__file__).with_name(name)) for name in ("parallel.py", "retrieval.py", "text.py", "data.py")},
        }
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["identity"] != identity:
                raise ValueError("Resume rejected: input, index, model, code, or chunk size changed")
        else:
            inventory = _scan_anchors(anchors_path)
            manifest = {"identity": identity, "anchors": inventory, "created_at": now()}
            atomic_json(manifest_path, manifest)
        total_rows = manifest["anchors"]["rows"]
        parts = output / "parts"
        parts.mkdir(exist_ok=True)
        state_path = output / "inference_progress.json"
        statistics = {"completed_rows": 0, "completed_chunks": 0, "candidate_pairs": 0, "matched_pairs": 0, "empty_matches": 0}
        worker_peaks = {}
        resumed_rows = 0
        last_progress = 0.0

        def add(metadata):
            statistics["completed_rows"] += metadata["rows"]
            statistics["completed_chunks"] += 1
            for key in ("candidate_pairs", "matched_pairs", "empty_matches"):
                statistics[key] += metadata[key]
            worker_peaks[str(metadata["worker_pid"])] = max(worker_peaks.get(str(metadata["worker_pid"]), 0), metadata["worker_peak_mib"])

        def progress(status, error=None):
            elapsed = time.monotonic() - started
            speed = (statistics["completed_rows"] - resumed_rows) / elapsed if elapsed else 0
            value = {"status": status, "updated_at": now(), "pid": os.getpid(), "workers": workers,
                     "total_rows": total_rows, **statistics, "resumed_rows": resumed_rows,
                     "elapsed_current_run_seconds": round(elapsed, 3), "rows_per_second": round(speed, 3),
                     "estimated_remaining_seconds": (total_rows - statistics["completed_rows"]) / speed if speed else None,
                     "worker_peak_mib": worker_peaks}
            if error:
                value["error"] = str(error)
            atomic_json(state_path, value)
            return value

        progress("running")
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = "1"
        iterator = iter(_chunks(anchors_path, chunk_size))
        exhausted = False
        dispatched = 0
        pending = {}
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                     initializer=_worker_init, initargs=(str(index.resolve()), str(model_dir.resolve()), str(parts.resolve()))) as pool:
                while pending or not exhausted:
                    while not exhausted and len(pending) < workers * 2:
                        try:
                            chunk_id, batch, chunk_digest = next(iterator)
                        except StopIteration:
                            exhausted = True
                            break
                        done = _completed_part(parts, chunk_id, batch, chunk_digest)
                        if done is not None:
                            add(done)
                            resumed_rows += done["rows"]
                            continue
                        if max_chunks is not None and dispatched >= max_chunks:
                            exhausted = True
                            break
                        pending[pool.submit(_worker_chunk, chunk_id, batch, chunk_digest)] = chunk_id
                        dispatched += 1
                    if not pending:
                        continue
                    done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                    for future in done:
                        pending.pop(future)
                        add(future.result())
                    if time.monotonic() - last_progress >= 5:
                        current = progress("running")
                        print(f"Predicted {current['completed_rows']:,}/{total_rows:,} anchors; {current['rows_per_second']:.1f}/s; {current['completed_chunks']:,} chunks saved", flush=True)
                        last_progress = time.monotonic()
            if statistics["completed_rows"] != total_rows:
                return progress("partial")
            if fingerprint(anchors_path, True) != identity["anchors"] or fingerprint(index) != identity["index"]:
                raise ValueError("Input or index changed while inference was running")
            progress("merging")
            for file_number, (name, header) in enumerate(zip(FILES, HEADERS)):
                temporary = output / (name + ".tmp")
                with temporary.open("wb") as stream:
                    stream.write(header.encode())
                    for chunk_id in range(statistics["completed_chunks"]):
                        paths, _ = _part_paths(parts, chunk_id)
                        with paths[file_number].open("rb") as part:
                            shutil.copyfileobj(part, stream, 1024 * 1024)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(output / name)
            complete = progress("complete")
            complete["output_files"] = {name: fingerprint(output / name, True) for name in FILES}
            atomic_json(output / "inference_complete.json", complete)
            print(json.dumps(complete, indent=2), flush=True)
            return complete
        except BaseException as error:
            progress("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error)
            raise
