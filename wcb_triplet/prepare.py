"""Step 1: prepare one WCB task, preserving the original split assignments."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "data/world_central_banks/all_annotated_sentences_25000"
LABEL_TO_ID = {"not forward looking": 0, "forward looking": 1}
SCHEMA = pa.schema([
    ("sentence_id", pa.string()),
    ("sentence", pa.string()),
    ("label", pa.int64()),
    ("time_label", pa.string()),
    ("bank_name", pa.string()),
    ("year", pa.int64()),
    ("source_split", pa.string()),
])


def text_key(text):
    # Used only to audit duplicates; preserve the input sentence in the output.
    return " ".join(text.split())


def prepare(seed, output):
    groups = defaultdict(list)
    report = {
        "task": "time_label",
        "seed": seed,
        "source": "https://huggingface.co/datasets/gtfintechlab/all_annotated_sentences_25000",
        "license": "CC-BY-NC-SA-4.0",
        "label_to_id": LABEL_TO_ID,
        "inputs": {},
        "deduplication": "Whitespace-equivalent text; retain test before val before train.",
        "document_split": "Unavailable: annotated source has no document ID; original WCB splits retained.",
    }
    source_ids = set()
    for split in ("train", "val", "test"):
        path = SOURCE_ROOT / seed / f"{split}-00000-of-00001.parquet"
        records = pq.read_table(path).to_pylist()
        report["inputs"][split] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(records),
        }
        for record in records:
            sentence = record["sentences"]
            label = record["time_label"]
            source_id = record["__index_level_0__"]
            if not isinstance(sentence, str) or not text_key(sentence):
                raise ValueError(f"Empty sentence: {split}, {source_id}")
            if label not in LABEL_TO_ID:
                raise ValueError(f"Unknown time label: {label!r}")
            if source_id in source_ids:
                raise ValueError(f"Source row ID appears multiple times: {source_id}")
            source_ids.add(source_id)
            groups[text_key(sentence)].append({
                "sentence_id": f"wcb:{source_id}",
                "sentence": sentence,
                "label": LABEL_TO_ID[label],
                "time_label": label,
                "bank_name": record["bank_name"],
                "year": record["year"],
                "source_split": split,
            })

    report["duplicate_text_groups"] = sum(len(rows) > 1 for rows in groups.values())
    report["cross_split_duplicate_text_groups"] = sum(
        len({r["source_split"] for r in rows}) > 1 for rows in groups.values()
    )
    report["conflicting_label_groups"] = sum(
        len({r["label"] for r in rows}) > 1 for rows in groups.values()
    )
    kept = {split: [] for split in ("train", "val", "test")}
    excluded = []
    priority = {"test": 0, "val": 1, "train": 2}
    for rows in groups.values():
        if len({r["label"] for r in rows}) > 1:
            for row in rows:
                excluded.append({**row, "reason": "conflicting_labels_for_same_text"})
            continue
        ordered = sorted(rows, key=lambda r: (priority[r["source_split"]], r["sentence_id"]))
        kept[ordered[0]["source_split"]].append(ordered[0])
        for row in ordered[1:]:
            excluded.append({**row, "reason": "duplicate_text"})

    output.mkdir(parents=True, exist_ok=True)
    report["outputs"] = {}
    seen_texts = set()
    for split, records in kept.items():
        records.sort(key=lambda r: r["sentence_id"])
        counts = Counter(r["time_label"] for r in records)
        if set(counts) != set(LABEL_TO_ID) or min(counts.values()) < 2:
            raise ValueError(f"{split} needs at least 2 sentences per time label")
        keys = {text_key(r["sentence"]) for r in records}
        if seen_texts & keys:
            raise AssertionError("Duplicate text survived across output splits")
        seen_texts.update(keys)
        pq.write_table(pa.Table.from_pylist(records, schema=SCHEMA), output / f"{split}.parquet")
        report["outputs"][split] = {"rows": len(records), "label_counts": dict(counts)}
    report["excluded_rows"] = len(excluded)
    with (output / "excluded.jsonl").open("w", encoding="utf-8") as stream:
        for row in excluded:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **report["outputs"],
                      "excluded_rows": len(excluded)}, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", default="5768", choices=["5768", "78516", "944601"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or PROJECT_ROOT / "data/processed" / f"wcb_time_{args.seed}"
    prepare(args.seed, output)


if __name__ == "__main__":
    main()
