"""Deterministic, class-balanced batches for supervised triplet learning."""

from collections import defaultdict
import math
from pathlib import Path
import random

import pyarrow.parquet as pq

from .prepare import LABEL_TO_ID, text_key


def load_rows(path: Path, split: str, samples_per_label: int | None, seed: int):
    rows = pq.read_table(path).to_pylist()
    groups = defaultdict(list)
    ids, texts = set(), set()
    for row in rows:
        if row["source_split"] != split:
            raise ValueError(f"Unexpected source split in {path}")
        if row["label"] != LABEL_TO_ID.get(row["time_label"]):
            raise ValueError(f"Invalid time label in {path}: {row['sentence_id']}")
        if not isinstance(row["sentence"], str) or not text_key(row["sentence"]):
            raise ValueError(f"Empty sentence in {path}")
        key = text_key(row["sentence"])
        if row["sentence_id"] in ids or key in texts:
            raise ValueError(f"Duplicate row or text in {path}")
        ids.add(row["sentence_id"])
        texts.add(key)
        groups[row["label"]].append(row)
    if set(groups) != {0, 1}:
        raise ValueError(f"Both time labels are required in {path}")
    if samples_per_label is not None:
        if samples_per_label < 2:
            raise ValueError("samples-per-label must be >=2")
        rng = random.Random(seed)
        selected = []
        for label in sorted(groups):
            if samples_per_label > len(groups[label]):
                raise ValueError(f"Not enough label {label} rows in {path}")
            selected.extend(rng.sample(groups[label], samples_per_label))
        rows = sorted(selected, key=lambda r: r["sentence_id"])
    return rows


def check_split_overlap(train_rows, val_rows):
    train_ids = {r["sentence_id"] for r in train_rows}
    train_texts = {text_key(r["sentence"]) for r in train_rows}
    if train_ids & {r["sentence_id"] for r in val_rows}:
        raise ValueError("Train and validation sentence IDs overlap")
    if train_texts & {text_key(r["sentence"]) for r in val_rows}:
        raise ValueError("Train and validation texts overlap")


class BalancedBatchSampler:
    """Each batch has equal class counts and no repeated row within the batch.

    An epoch covers every row in the largest class. The smaller class, and the
    final partial batch, are filled by deterministic shuffled cycling. This is
    oversampling, not an increase in the number of independent observations.
    """

    def __init__(self, labels, batch_size: int, seed: int):
        self.groups = defaultdict(list)
        for index, label in enumerate(labels):
            self.groups[label].append(index)
        if set(self.groups) != {0, 1}:
            raise ValueError("Both time classes are required")
        if batch_size < 4 or batch_size % 2:
            raise ValueError("batch-size must be even and >=4")
        self.per_class = batch_size // 2
        if min(map(len, self.groups.values())) < self.per_class:
            raise ValueError("Each class must contain at least batch-size / 2 unique rows")
        self.seed = seed
        self.num_batches = math.ceil(max(map(len, self.groups.values())) / self.per_class)

    def __len__(self):
        return self.num_batches

    def batches(self, epoch: int):
        rng = random.Random(self.seed + 1_000_003 * epoch)
        pools = {label: [] for label in self.groups}
        result = []
        for _ in range(self.num_batches):
            batch = []
            for label in sorted(self.groups):
                selected = set()
                while len(selected) < self.per_class:
                    if not pools[label]:
                        pools[label] = list(self.groups[label])
                        rng.shuffle(pools[label])
                    index = pools[label].pop()
                    if index not in selected:
                        selected.add(index)
                        batch.append(index)
            rng.shuffle(batch)
            result.append(batch)
        return result
