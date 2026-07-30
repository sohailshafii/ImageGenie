"""Deterministic train/val/test splits (M6 B5, ml.md#dataset-splits).

Every model's partition is a pure function of **its own uid and the seed** —
nothing else. Not the labels, not the corpus size, not what any other model is
called. That property is the whole point, and it was learned the expensive way:
the previous implementation shuffled each class with one shared RNG and sliced,
which made the partition a function of the *entire* labeled set. Correcting a
single label then re-randomised most of the corpus (measured: one change in 3,600
moved 127 models in or out of test), because the class it left was one shorter, so
its shuffle consumed a different number of draws and every class after it in sort
order got a different permutation too.

That made the milestone-8 loop unable to measure itself: run 14 and run 15
straddled a 24-label correction pass and ended up sharing only 29% of their test
sets, so scoring each on its own split said the retrain was 4.5 points worse while
scoring both on the models they both held out said it was 5.3 points better
(ml.md#evaluation). Hashing per uid means a corrected label moves a model between
*classes* but never between *train and test*, so two runs across a correction pass
are comparable by construction.

The trade is that per-class proportions are now approximate rather than exact —
each class lands near 80/10/10 by the law of large numbers instead of on it. At
the roster's ~300+ members per class that drift is a handful of models, it is
recorded in every run's data snapshot, and it buys a partition that no relabelling
can disturb.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass

# One sample = (model uid, class name). Kept as a plain tuple to match what the
# dataset and the label query pass around.
Sample = tuple[str, str]

# Hash buckets per model. 10_000 rather than 100 so a 10% slice is exact to a
# basis point, which keeps small classes from drifting on coarse rounding.
BUCKET_COUNT = 10_000


@dataclass(frozen=True)
class DatasetSplit:
    """The three disjoint partitions, each a list of (uid, class_name)."""

    train: list[Sample]
    val: list[Sample]
    test: list[Sample]


def bucket_of(uid: str, seed: int) -> int:
    """A stable bucket in ``[0, BUCKET_COUNT)`` for one model.

    sha256 rather than `hash()`: the builtin is randomised per process unless
    PYTHONHASHSEED is pinned, which would make a run's partition depend on the
    interpreter that produced it — reproducibility (NFR-4) that evaporates
    silently on the next process.
    """
    digest = hashlib.sha256(f"{seed}:{uid}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % BUCKET_COUNT


def stratified_split(
    samples: list[Sample],
    seed: int,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> DatasetSplit:
    """Partition ``samples`` into train/val/test by hashing each uid.

    Still stratified in the sense that matters — the hash is independent of class,
    so each class lands in each partition at the same expected rate, and no class
    can be absent from train by construction. What it no longer does is *force*
    exact per-class counts, which is what tied the partition to the label set.

    A class too small to fill a val/test slot is not special-cased: with ~300+
    members per roster class the buckets fill naturally, and a tiny local-smoke
    class simply gets whatever its uids hash to.
    """
    test_cutoff = int(BUCKET_COUNT * test_fraction)
    val_cutoff = test_cutoff + int(BUCKET_COUNT * val_fraction)

    partitions: dict[str, list[Sample]] = defaultdict(list)
    # Sorted so the output order is deterministic regardless of input order; the
    # *assignment* already is, but a stable order keeps runs byte-comparable.
    for uid, class_name in sorted(samples):
        bucket = bucket_of(uid, seed)
        if bucket < test_cutoff:
            partitions["test"].append((uid, class_name))
        elif bucket < val_cutoff:
            partitions["val"].append((uid, class_name))
        else:
            partitions["train"].append((uid, class_name))

    return DatasetSplit(
        train=partitions["train"], val=partitions["val"], test=partitions["test"]
    )


def split_sizes(split: DatasetSplit) -> dict[str, int]:
    """Per-partition counts — recorded in the run's data snapshot (NFR-4)."""
    return {"train": len(split.train), "val": len(split.val), "test": len(split.test)}
