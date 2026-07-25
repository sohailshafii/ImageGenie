"""Deterministic stratified train/val/test splits (M6 B5, ml.md#dataset-splits).

The labeled set is partitioned **per class** so every class appears in every
split (stratified), at ~80/10/10. The split is a pure function of the sample set
and a seed, so a run is reproducible (NFR-4): same samples + same seed → the same
partition, and the sizes are recorded in the run's data snapshot.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

# One sample = (model uid, class name). Kept as a plain tuple to match what the
# dataset and the label query pass around.
Sample = tuple[str, str]


@dataclass(frozen=True)
class DatasetSplit:
    """The three disjoint partitions, each a list of (uid, class_name)."""

    train: list[Sample]
    val: list[Sample]
    test: list[Sample]


def stratified_split(
    samples: list[Sample],
    seed: int,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> DatasetSplit:
    """Partition ``samples`` per class into train/val/test.

    Deterministic: within each class the uids are sorted (so input order does not
    matter) then shuffled with a seeded RNG, then sliced. ``floor`` is used for the
    val/test counts so train keeps the remainder — a class too small to fill a
    val/test slot puts all of its members in train rather than starving train (the
    roster classes clear ~300 on the seed, so this only bites tiny local smokes).
    """
    class_to_uids: dict[str, list[str]] = defaultdict(list)
    for uid, class_name in samples:
        class_to_uids[class_name].append(uid)

    random_source = random.Random(seed)
    train: list[Sample] = []
    val: list[Sample] = []
    test: list[Sample] = []
    # sorted() so the RNG is consumed in a class order that doesn't depend on dict
    # insertion — part of what makes the split reproducible.
    for class_name in sorted(class_to_uids):
        uids = sorted(class_to_uids[class_name])
        random_source.shuffle(uids)
        count = len(uids)
        test_count = int(count * test_fraction)
        val_count = int(count * val_fraction)
        test.extend((uid, class_name) for uid in uids[:test_count])
        val.extend((uid, class_name) for uid in uids[test_count : test_count + val_count])
        train.extend((uid, class_name) for uid in uids[test_count + val_count :])
    return DatasetSplit(train=train, val=val, test=test)


def split_sizes(split: DatasetSplit) -> dict[str, int]:
    """Per-partition counts — recorded in the run's data snapshot (NFR-4)."""
    return {"train": len(split.train), "val": len(split.val), "test": len(split.test)}
