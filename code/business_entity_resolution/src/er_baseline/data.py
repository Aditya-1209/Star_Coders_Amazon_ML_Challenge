import csv
from dataclasses import dataclass
from pathlib import Path

SOURCE_HEADER = ["entity_id", "business_name", "business_address", "country"]
TRUTH_HEADER = ["source1_entity_id", "matched_entity_ids"]


@dataclass(frozen=True)
class Record:
    entity_id: str
    business_name: str
    business_address: str
    country: str

    def values(self):
        return [self.entity_id, self.business_name, self.business_address, self.country]


def read_tsv(path, expected):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        header = next(reader, None)
        if header != expected:
            raise ValueError(f"Wrong header in {path}: {header!r}")
        for row in reader:
            if len(row) != len(expected):
                raise ValueError(f"Malformed TSV row in {path}, line {reader.line_num}")
            yield row


def read_records(path):
    for row in read_tsv(path, SOURCE_HEADER):
        yield Record(*row)


def read_truth(path):
    result = {}
    for entity_id, text in read_tsv(path, TRUTH_HEADER):
        if entity_id in result:
            raise ValueError(f"Duplicate Source 1 label: {entity_id}")
        ids = text.split(",") if text else []
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate target label for {entity_id}")
        result[entity_id] = set(ids)
    return result


def write_tsv(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)

