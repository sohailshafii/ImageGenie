"""Load weak labels from ``weak_labels.csv`` into the ``label`` table.

The labeling UI reads its classes from the DB, but weak labeling (FR-3) writes a
CSV — so without this step every model shows as "unlabeled" and there is nothing
for an admin to confirm or correct. This is the bridge, run once after ingestion
and again whenever the weak-labeling rules are re-run.

    python -m app.backfill_labels --labels data/exploration/weak_labels.csv

**Confidence is the measured per-class precision** from ``weak_label_eval.json``
(``ml/eval_weak_labels.py``), not a made-up number: it is literally "how often is
a weak label of this class correct", graded against the LVIS gold set. That makes
lowest-confidence-first a meaningful review order in the UI — ``figure`` (0.62,
the known figure/animal boundary) surfaces ahead of ``lamp`` (1.00) — and it is
the ordering the active-learning loop (milestone 8) wants.

**Idempotent (NFR-2):** a model that already has a weak label is skipped, so
re-running never duplicates rows. Manual corrections are untouched — they are
separate rows, and the API resolves the most recent label as current, so a
re-import can't clobber human work.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from sqlalchemy import select

from .db import init_db, session_scope
from .models import Label, LabelSource, Model

logger = logging.getLogger(__name__)

INSERT_BATCH_SIZE = 1000


def load_class_confidence(eval_path: Path) -> dict[str, float]:
    """Map each class to its measured precision, used as the label confidence."""
    metrics = json.loads(eval_path.read_text(encoding="utf-8"))["per_class_metrics"]
    return {
        class_name: row["precision"]
        for class_name, row in metrics.items()
        if row.get("precision") is not None
    }


def read_weak_labels(labels_path: Path) -> list[tuple[str, str]]:
    """Read ``(uid, class_name)`` pairs from a ``uid,class,reason`` CSV."""
    with labels_path.open(newline="", encoding="utf-8") as csv_file:
        return [(row["uid"], row["class"]) for row in csv.DictReader(csv_file)]


def backfill(labels_path: Path, eval_path: Path | None, dry_run: bool) -> dict[str, int]:
    """Insert a weak `label` row per uid that has a model and no weak label yet.

    Returns a count breakdown so the caller can report what actually happened —
    "skipped" is the interesting number on a rerun.
    """
    rows = read_weak_labels(labels_path)
    class_to_confidence: dict[str, float] = {}
    if eval_path is not None and eval_path.exists():
        class_to_confidence = load_class_confidence(eval_path)
    else:
        logger.warning(
            "no eval file at %s — inserting labels with NULL confidence, so the UI "
            "cannot order by review priority",
            eval_path,
        )

    unmeasured_classes_set = {class_name for _, class_name in rows} - set(class_to_confidence)
    if class_to_confidence and unmeasured_classes_set:
        # A class in the CSV that the eval never scored means the two artifacts
        # came from different runs — worth saying out loud rather than silently
        # writing NULL confidences for part of the set.
        logger.warning(
            "%d class(es) in the CSV have no measured precision: %s",
            len(unmeasured_classes_set),
            ", ".join(sorted(unmeasured_classes_set)),
        )

    with session_scope() as session:
        known_uids_set = set(session.scalars(select(Model.uid)).all())
        already_weak_uids_set = set(
            session.scalars(
                select(Label.model_uid).where(Label.source == LabelSource.weak)
            ).all()
        )

        pending: list[Label] = []
        counts = {"total": len(rows), "missing_model": 0, "skipped": 0, "inserted": 0}
        for uid, class_name in rows:
            if uid not in known_uids_set:
                # Expected, not an error: the CSV covers the whole labeled set,
                # while the DB only has what actually finished downloading.
                counts["missing_model"] += 1
                continue
            if uid in already_weak_uids_set:
                counts["skipped"] += 1
                continue
            pending.append(
                Label(
                    model_uid=uid,
                    class_name=class_name,
                    source=LabelSource.weak,
                    confidence=class_to_confidence.get(class_name),
                    annotator=None,  # rule-derived, no human behind it
                )
            )
            already_weak_uids_set.add(uid)  # guard against duplicate CSV rows
            counts["inserted"] += 1

        if dry_run:
            logger.info("dry run — no rows written")
            return counts

        for start in range(0, len(pending), INSERT_BATCH_SIZE):
            session.add_all(pending[start : start + INSERT_BATCH_SIZE])
            session.flush()  # bound memory; the whole run stays one transaction
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/exploration/weak_labels.csv"),
        help="weak_labels.csv produced by ml/weak_label.py",
    )
    parser.add_argument(
        "--eval",
        type=Path,
        default=Path("data/exploration/weak_label_eval.json"),
        help="weak_label_eval.json — supplies per-class precision as confidence",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would be written, write nothing"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    init_db()
    counts = backfill(args.labels, args.eval, args.dry_run)
    logger.info(
        "%d CSV rows: %d inserted, %d already had a weak label, %d had no model row",
        counts["total"],
        counts["inserted"],
        counts["skipped"],
        counts["missing_model"],
    )


if __name__ == "__main__":
    main()
