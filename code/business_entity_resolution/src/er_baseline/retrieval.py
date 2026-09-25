"""Disk-backed, country-specific BM25 retrieval with word and trigram channels."""

from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
import sqlite3
import time

from .data import Record, read_records
from .text import ADDRESS_COMMON, LEGAL, address_normalize, grams, name_core, normalize


INDEX_VERSION = 2


def build_index(paths, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite index: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    db = sqlite3.connect(destination)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA cache_size=-131072")
    db.execute("CREATE TABLE records (rowid INTEGER PRIMARY KEY, entity_id TEXT UNIQUE NOT NULL, business_name TEXT, business_address TEXT, country TEXT)")
    db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    countries, counts = {}, defaultdict(int)
    rowid = 0
    try:
        for path in paths:
            for record in read_records(path):
                if not record.entity_id.startswith(("S2-", "S3-")):
                    raise ValueError(f"Invalid target ID {record.entity_id}")
                country = normalize(record.country)
                if country not in countries:
                    table = f"fts_{len(countries)}"
                    countries[country] = table
                    # BM25 needs term occurrence positions for contentless FTS.
                    # detail=column makes every BM25 score zero in this setup.
                    db.execute(f"CREATE VIRTUAL TABLE {table} USING fts5(name, address, grams, content='', detail=full)")
                table = countries[country]
                rowid += 1
                db.execute("INSERT INTO records VALUES (?,?,?,?,?)", (rowid, *record.values()))
                name = name_core(normalize(record.business_name))
                address = address_normalize(record.business_address)
                db.execute(f"INSERT INTO {table}(rowid,name,address,grams) VALUES (?,?,?,?)", (rowid, name, address, " ".join(grams(name))))
                counts[country] += 1
                if rowid % 10000 == 0:
                    db.commit()
                if rowid % 100000 == 0:
                    print(f"Indexed {rowid:,} target records", flush=True)
        for table in countries.values():
            db.execute(f"INSERT INTO {table}({table}) VALUES ('optimize')")
            db.execute(f"CREATE VIRTUAL TABLE {table}_vocab USING fts5vocab({table}, 'col')")
        metadata = {"version": INDEX_VERSION, "countries": countries, "counts": dict(counts), "records": rowid, "source_paths": [str(Path(p).resolve()) for p in paths], "build_seconds": round(time.monotonic() - started, 3)}
        db.executemany("INSERT INTO metadata VALUES (?,?)", ((key, json.dumps(value)) for key, value in metadata.items()))
        db.commit()
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        print(json.dumps(metadata, indent=2), flush=True)
        return metadata
    finally:
        db.close()


class Retriever:
    def __init__(self, path, per_channel=20, posting_budget=0):
        self.db = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
        self.db.execute("PRAGMA cache_size=-65536")
        self.metadata = {key: json.loads(value) for key, value in self.db.execute("SELECT key,value FROM metadata")}
        if self.metadata.get("version") != INDEX_VERSION:
            raise ValueError("Unsupported index version")
        self.countries = self.metadata["countries"]
        self.per_channel = per_channel
        self.posting_budget = posting_budget

    @lru_cache(maxsize=100000)
    def frequency(self, table, column, token):
        row = self.db.execute(f"SELECT doc FROM {table}_vocab WHERE term=? AND col=?", (token, column)).fetchone()
        return row[0] if row else 0

    def query(self, record):
        table = self.countries.get(normalize(record.country))
        if table is None:
            return []
        core = name_core(normalize(record.business_name))
        name_terms = set(core.split()) - LEGAL
        address_terms = set(address_normalize(record.business_address).split()) - ADDRESS_COMMON
        requests = [("name", name_terms, 6), ("address", address_terms, 7), ("grams", grams(core), 10)]
        ranks = {}
        for channel, (column, tokens, term_limit) in enumerate(requests):
            # Use the rarest terms that actually occur in the index. This limits
            # huge common-word posting lists while tolerating missing components.
            ranked = [(self.frequency(table, column, token), token) for token in tokens]
            ranked = [(count, token) for count, token in sorted(ranked) if count > 0][:term_limit]
            terms = [token for count, token in ranked]
            if not terms:
                continue
            all_terms = list(terms)
            operator = " OR "
            if self.posting_budget:
                remaining = self.posting_budget
                terms = []
                for count, token in ranked:
                    if count <= remaining:
                        terms.append(token)
                        remaining -= count
                if not terms:
                    # All terms are common: intersect the two rarest instead of
                    # scoring the union of millions of generic-name records.
                    terms = [token for _, token in ranked[:2]]
                    operator = " AND "
            expression = column + " : (" + operator.join('"' + token.replace('"', '""') + '"' for token in terms) + ")"
            if self.posting_budget and terms != all_terms:
                # Keep the other query terms in BM25 scoring, while using rare
                # terms as a gate. Dropping their scores loses multiword matches.
                scoring = column + " : (" + " OR ".join('"' + token.replace('"', '""') + '"' for token in all_terms) + ")"
                expression = "(" + scoring + ") AND (" + expression + ")"
            found = self.db.execute(f"SELECT rowid FROM {table} WHERE {table} MATCH ? ORDER BY rank LIMIT ?", (expression, self.per_channel))
            for rank, (rowid,) in enumerate(found, 1):
                ranks.setdefault(rowid, [0, 0, 0])[channel] = rank
        if not ranks:
            return []
        placeholders = ",".join("?" for _ in ranks)
        records = self.db.execute(f"SELECT rowid,entity_id,business_name,business_address,country FROM records WHERE rowid IN ({placeholders})", tuple(ranks))
        result = [(Record(*row[1:]), ranks[row[0]]) for row in records]
        return sorted(result, key=lambda item: (-sum(1 / rank for rank in item[1] if rank), item[0].entity_id))

    def close(self):
        self.db.close()
        self.frequency.cache_clear()
