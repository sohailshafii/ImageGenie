"""Select the second dev set — LVIS gold objects we have *not* ingested (FR-7, M8).

The first dev set is the `test` slice of our own corpus, and it inherits our own
weak labels, so it can only ever measure the model against the label noise that
trained it. FR-7 asks for a second, and milestone 8 needs one for a sharper
reason: a reviewer who has already seen the classifier's guess cannot produce an
unbiased correction, so the M8 accuracy gain is a biased estimator by
construction. LVIS annotations were made without reference to this model at all,
which is the property no amount of in-house relabeling can buy.

Only ~475 LVIS gold objects are already in our trainable set and just ~49 of
those land in `test` — far too thin to report, and scoring the rest is
meaningless because the model trained on them. So the second dev set has to be
objects we have never ingested: this script picks them, and the existing
(idempotent) M4 pipeline ingests them.

    make devset                       # 1,000 candidates, balanced across the roster
    make devset DEVSET_COUNT=200      # a pilot

Writes ``uid,class,reason`` — the same shape ``weak_labels.csv`` has — so the
selection feeds the seeder unchanged:

    python -m app.seed --from-labels data/devset/lvis_dev.csv --count 1000

**The class column is deliberately not loaded into the `label` table.** A model
is trainable exactly when it is labeled *and* rendered (`train.load_trainable_
samples`), so leaving these unlabeled makes them structurally impossible to train
on — the dev set cannot leak into a future run through an absent-minded backfill.
The labels live here, in a file, and the evaluator reads them from here.

Requires the DB (to know what is already ingested): run it against Cloud SQL
through the proxy, as `ml/evaluate.py` is run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path

from eval_weak_labels import build_uid_to_gold_class
from io_utils import write_csv
from sqlalchemy import select
from taxonomy import ROSTER

from app.db import session_scope
from app.models import Model

# Where the selection lands, and where the evaluator reads it back from. One
# constant because the two ends must agree on a path no migration guards: the
# labels live in a file precisely so they never reach the `label` table.
DEV_SET_PATH = Path("data/devset/lvis_dev.csv")


def load_dev_set(path: Path = DEV_SET_PATH) -> list[tuple[str, str]]:
    """Read the selection back as (uid, gold class) pairs.

    Rows outside the roster are dropped rather than raising, matching
    `train.load_trainable_samples`: an unknown class name would otherwise surface
    as a KeyError deep inside a DataLoader worker, and one stray row should not
    lose a scoring pass. The count is printed so it cannot pass silently.
    """
    if not path.exists():
        raise SystemExit(
            f"{path} not found — run `make devset` to build the selection. "
            "It is gitignored (NFR-6), so a fresh checkout has to regenerate it; "
            "the hash ordering makes that reproduce the same objects."
        )
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = [(row["uid"], row["class"]) for row in csv.DictReader(csv_file)]

    samples = [(uid, class_name) for uid, class_name in rows if class_name in ROSTER]
    dropped = len(rows) - len(samples)
    if dropped:
        print(f"skipped {dropped} dev-set row(s) outside the roster")
    return samples


def load_ingested_uids() -> set[str]:
    """Every uid the pipeline has already seen, soft-deleted ones included.

    Soft-deleted models are deliberately in scope: they were excluded from the
    corpus on purpose, and re-seeding them would both re-download bytes we chose
    to drop and quietly resurrect a decision. "Not ingested" here means "no row
    in `model`", not "no usable model".
    """
    with session_scope() as session:
        return set(session.scalars(select(Model.uid)).all())


def hash_order(uid: str) -> str:
    """A stable, seed-free ordering key over uids.

    Sorting by `sha256(uid)` rather than shuffling means the selection depends on
    nothing but the uids themselves — rerun it after ingesting more models and
    the objects it already picked keep their relative order, so the dev set grows
    rather than reshuffling. That is the same property `ml/splits.py` buys with
    the same trick, and for the same reason (ml.md#why-the-split-is-hashed-not-shuffled).
    """
    return hashlib.sha256(uid.encode()).hexdigest()


def select_dev_set(
    uid_to_gold_class: dict[str, str], ingested_uids_set: set[str], total: int
) -> list[tuple[str, str]]:
    """Pick `total` un-ingested gold objects, as evenly across the roster as supply allows.

    Balanced rather than proportional on purpose. The corpus is skewed ~7.7:1 and
    the tail classes are where the model actually fails, so a dev set that mirrors
    the skew would spend most of its budget re-confirming what `weapon` already
    tells us. Classes short of their quota contribute everything they have and the
    shortfall is redistributed, so a thin class costs coverage, not the whole budget.
    """
    class_to_candidates: dict[str, list[str]] = {class_name: [] for class_name in ROSTER}
    for uid, class_name in uid_to_gold_class.items():
        if uid not in ingested_uids_set and class_name in class_to_candidates:
            class_to_candidates[class_name].append(uid)
    for candidates in class_to_candidates.values():
        candidates.sort(key=hash_order)

    quota = total // len(ROSTER)
    selected: list[tuple[str, str]] = []
    leftovers: list[tuple[str, str]] = []
    for class_name in ROSTER:
        candidates = class_to_candidates[class_name]
        selected.extend((uid, class_name) for uid in candidates[:quota])
        leftovers.extend((uid, class_name) for uid in candidates[quota:])

    # Redistribute the shortfall from classes that could not fill their quota,
    # in the same hash order so the top-up is as reproducible as the quota.
    leftovers.sort(key=lambda entry: hash_order(entry[0]))
    selected.extend(leftovers[: max(0, total - len(selected))])
    selected.sort(key=lambda entry: hash_order(entry[0]))
    return selected


def count_weak_label_overlap(selected: list[tuple[str, str]], labels_path: Path) -> int:
    """How many selected uids also appear in `weak_labels.csv`.

    Worth knowing, not worth excluding on. `make backfill-labels` loads that file
    into the `label` table, so an overlapping uid would acquire a weak label once
    ingested and become trainable — the one route by which this dev set could end
    up in a training run. The evaluator asserts non-overlap with the run's
    training samples anyway, so this is an early warning rather than the guard.
    """
    if not labels_path.exists():
        print(f"note: {labels_path} not present — skipping the overlap check")
        return 0
    with labels_path.open(newline="", encoding="utf-8") as csv_file:
        weak_labeled_uids_set = {row["uid"] for row in csv.DictReader(csv_file)}
    return sum(1 for uid, _ in selected if uid in weak_labeled_uids_set)


def report(
    uid_to_gold_class: dict[str, str],
    ingested_uids_set: set[str],
    selected: list[tuple[str, str]],
) -> None:
    """Print per-class supply and selection, so a thin class is visible before the data run."""
    gold_counts = Counter(uid_to_gold_class.values())
    available_counts = Counter(
        class_name
        for uid, class_name in uid_to_gold_class.items()
        if uid not in ingested_uids_set
    )
    selected_counts = Counter(class_name for _, class_name in selected)

    print(f"\n{'class':<14}{'gold':>8}{'un-ingested':>14}{'selected':>10}")
    for class_name in ROSTER:
        print(
            f"{class_name:<14}{gold_counts[class_name]:>8,}"
            f"{available_counts[class_name]:>14,}{selected_counts[class_name]:>10,}"
        )
    print(
        f"\n{len(uid_to_gold_class):,} LVIS gold objects in the roster; "
        f"{len(ingested_uids_set):,} uids already ingested; "
        f"{sum(available_counts.values()):,} gold objects available; "
        f"{len(selected):,} selected."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the second dev set (FR-7).")
    parser.add_argument("--count", type=int, default=1000,
                        help="how many objects to select (default: 1,000)")
    parser.add_argument("--out", type=Path, default=DEV_SET_PATH,
                        help="where to write the uid,class,reason CSV (gitignored)")
    parser.add_argument("--weak-labels", type=Path,
                        default=Path("data/exploration/weak_labels.csv"),
                        help="weak-label CSV to check for overlap (see module docstring)")
    args = parser.parse_args()

    uid_to_gold_class = build_uid_to_gold_class()
    ingested_uids_set = load_ingested_uids()
    selected = select_dev_set(uid_to_gold_class, ingested_uids_set, args.count)
    report(uid_to_gold_class, ingested_uids_set, selected)

    overlap = count_weak_label_overlap(selected, args.weak_labels)
    if overlap:
        print(
            f"\nWARNING: {overlap:,} of the selected uids also appear in "
            f"{args.weak_labels}. `make backfill-labels` would label them, which "
            "makes them trainable — keep this dev set out of that backfill."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.out,
        ("uid", "class", "reason"),
        [(uid, class_name, "lvis-gold") for uid, class_name in selected],
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
