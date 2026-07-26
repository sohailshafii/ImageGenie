"""Delete raw source meshes for models excluded from the dataset.

CLAUDE.md's cost guardrail: "Delete raw files for models excluded from the
dataset." The raw bucket is the bulk of the storage line, and a mesh that will
never be trained on is paying rent for nothing.

**Deliberately narrow.** It deletes raw *only* for models that are not in the
dataset at all — soft-deleted, unlabeled, or labeled outside the 12-class roster
— plus stray blobs with no model row. Every model still in the dataset keeps its
source mesh, so any stage can be re-run from scratch. It never touches
``processed/``.

**Only raw is removable in the first place.** The processed artifacts are what
the labeling UI and training read; raw is read once, by convert. Deleting raw is
also compatible with the reconciler, which marks a model whose mesh is gone as
``pending`` rather than dropping it (server.md#rebuilding-the-tables-from-storage).

    python -m app.cleanup_raw                 # dry run: report, delete nothing
    python -m app.cleanup_raw --apply         # delete, after a typed confirmation

`make cleanup-raw` wraps the dry run; `make cleanup-raw APPLY=1` the real thing.

**This is an operator tool, not worker-image code.** It is the one module under
``app/`` that imports from ``ml/`` (the class roster, whose single source of
truth is ``ml/taxonomy.py``), which is why the Makefile target puts both on
PYTHONPATH — mirroring how `make train` puts ``server/`` on the path for
``ml/train.py``. Nothing in the worker image imports this module.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session
from taxonomy import ROSTER

from .artifact_keys import RAW_PREFIX, uid_from_key
from .db import session_scope
from .models import Label, Model
from .storage import Storage, build_storage

logger = logging.getLogger(__name__)

# Why a given mesh is not in the dataset. Reported per-reason so a dry run shows
# *what kind* of exclusion is reclaiming the space, not just a total — a surprise
# in the mix (say, thousands "unlabeled") is the signal to stop and look rather
# than pass --apply.
REASON_ORPHAN = "no model row"
REASON_DELETED = "soft-deleted"
REASON_UNLABELED = "no label"
REASON_OUT_OF_SCOPE = "label outside the roster"


@dataclass(frozen=True)
class Candidate:
    """One raw blob that would be deleted, and why."""

    key: str
    uid: str | None  # None only for a key that parses to no uid
    reason: str
    size_bytes: int


def _excluded_uids(session: Session) -> tuple[set[str], dict[str, str]]:
    """(soft-deleted uids, uid → current class name) for every model row.

    The current label is the most recent one, manual beating weak — the same
    resolution the API uses to answer "what is this model labeled?", so the
    cleanup and the UI can never disagree about what is in the dataset.
    """
    deleted_uids_set = set(
        session.scalars(select(Model.uid).where(Model.deleted_at.is_not(None))).all()
    )
    current_labels = (
        select(Label.model_uid, Label.class_name)
        .distinct(Label.model_uid)
        .order_by(Label.model_uid, Label.created_at.desc(), Label.id.desc())
    )
    uid_to_class_name = {
        model_uid: class_name for model_uid, class_name in session.execute(current_labels)
    }
    return deleted_uids_set, uid_to_class_name


def _classify(
    key: str,
    size_bytes: int,
    known_uids_set: set[str],
    deleted_uids_set: set[str],
    uid_to_class_name: dict[str, str],
    in_scope_set: frozenset[str],
) -> Candidate | None:
    """The exclusion reason for one raw key, or None to keep it."""
    uid = uid_from_key(key)
    if uid is None:
        # An unrecognised key under raw/. Reported, never deleted: the reconciler
        # treats stray objects the same way, and a key we cannot parse is one we
        # cannot claim to own.
        return None
    if uid not in known_uids_set:
        return Candidate(key, uid, REASON_ORPHAN, size_bytes)
    if uid in deleted_uids_set:
        return Candidate(key, uid, REASON_DELETED, size_bytes)
    class_name = uid_to_class_name.get(uid)
    if class_name is None:
        return Candidate(key, uid, REASON_UNLABELED, size_bytes)
    if class_name not in in_scope_set:
        return Candidate(key, uid, REASON_OUT_OF_SCOPE, size_bytes)
    return None


def select_candidates(
    session: Session,
    storage: Storage,
    in_scope: Iterable[str] = ROSTER,
) -> Iterator[Candidate]:
    """Stream the raw blobs that are excluded from the dataset.

    One listing pass over ``raw/`` joined against two in-memory maps of the model
    table. The maps are loaded up front rather than queried per blob: ~12k models
    is a few MB, where a query per object would be ~165k round-trips.

    `in_scope` is a parameter rather than a constant so a test can pin behaviour
    without depending on the live roster.
    """
    in_scope_set = frozenset(in_scope)
    deleted_uids_set, uid_to_class_name = _excluded_uids(session)
    known_uids_set = set(session.scalars(select(Model.uid)).all())

    for key, size_bytes in storage.list_sizes(RAW_PREFIX):
        candidate = _classify(
            key,
            size_bytes,
            known_uids_set,
            deleted_uids_set,
            uid_to_class_name,
            in_scope_set,
        )
        if candidate is not None:
            yield candidate


def summarize(candidates: Iterable[Candidate]) -> tuple[Counter, Counter]:
    """(count per reason, bytes per reason) — what the dry run prints."""
    counts: Counter = Counter()
    total_bytes: Counter = Counter()
    for candidate in candidates:
        counts[candidate.reason] += 1
        total_bytes[candidate.reason] += candidate.size_bytes
    return counts, total_bytes


def human_bytes(size: int) -> str:
    """`1536` → `1.5 KB`. For the report only; never used in a comparison."""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{int(value)} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"  # unreachable; keeps the return type total


CONFIRMATION_PHRASE = "delete excluded raw meshes"

# How often the delete loop reports. Deleting is one request per object, so a
# large run is minutes long — silence for that stretch is indistinguishable from
# a hang.
_PROGRESS_EVERY = 250


def delete_candidates(storage: Storage, candidates: Iterable[Candidate]) -> tuple[int, int]:
    """Delete each candidate blob; return (objects deleted, bytes reclaimed).

    Deletion is per object rather than a prefix wipe, because the excluded set is
    scattered through ``raw/`` — the keeping and the deleting live side by side
    under one prefix, so there is no prefix that selects only the excluded.

    A failure on one object is logged and skipped rather than aborting: the run
    is idempotent, so finishing the rest and re-running beats stopping halfway
    with no record of where.
    """
    deleted_count = 0
    reclaimed_bytes = 0
    for candidate in candidates:
        try:
            storage.delete(candidate.key)
        except Exception:
            logger.warning("could not delete %s — skipping", candidate.key, exc_info=True)
            continue
        deleted_count += 1
        reclaimed_bytes += candidate.size_bytes
        if deleted_count % _PROGRESS_EVERY == 0:
            logger.info(
                "deleted %d objects (%s)", deleted_count, human_bytes(reclaimed_bytes)
            )
    return deleted_count, reclaimed_bytes


def _report(counts: Counter, total_bytes: Counter) -> int:
    """Print the per-reason table; return the total bytes it accounts for."""
    logger.info("%-28s %8s %12s", "reason", "objects", "size")
    for reason in (REASON_ORPHAN, REASON_DELETED, REASON_UNLABELED, REASON_OUT_OF_SCOPE):
        logger.info(
            "%-28s %8d %12s", reason, counts[reason], human_bytes(total_bytes[reason])
        )
    total = sum(total_bytes.values())
    logger.info("%-28s %8d %12s", "TOTAL", sum(counts.values()), human_bytes(total))
    return total


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually delete (default is a dry run that changes nothing)",
    )
    args = parser.parse_args()

    from .config import get_settings

    storage = build_storage(get_settings())
    # Materialised, not streamed: the report has to be shown and confirmed
    # *before* anything is deleted, and re-listing afterwards would race the
    # pipeline — a model labeled between the two passes would be deleted without
    # having appeared in the table the operator agreed to.
    with session_scope() as session:
        candidates = list(select_candidates(session, storage))

    counts, total_bytes = summarize(candidates)
    total = _report(counts, total_bytes)

    if not candidates:
        logger.info("nothing to delete — no excluded raw meshes.")
        return

    if not args.apply:
        logger.info(
            "\nDry run — nothing was deleted. Re-run with --apply to reclaim %s.\n"
            "Restoring these meshes afterwards means re-downloading them from "
            "Objaverse; the renders cannot rebuild them.",
            human_bytes(total),
        )
        return

    logger.warning(
        "\nAbout to PERMANENTLY DELETE %d raw meshes (%s) from the raw bucket.\n"
        "This is the only copy — restoring them means re-downloading from Objaverse.",
        len(candidates),
        human_bytes(total),
    )
    reply = input(f"  Type '{CONFIRMATION_PHRASE}' to proceed: ")
    if reply.strip() != CONFIRMATION_PHRASE:
        logger.error("aborted — nothing was deleted.")
        raise SystemExit(1)

    deleted_count, reclaimed_bytes = delete_candidates(storage, candidates)
    logger.info("deleted %d objects, reclaimed %s", deleted_count, human_bytes(reclaimed_bytes))
    # The `model` rows are deliberately left alone: object storage is the durable
    # record and the rows are an index over it, so the next `make reconcile-storage`
    # will mark these `pending` on its own (server.md#rebuilding-the-tables-from-storage).
    # Clearing `download_status` here would be worse than leaving it — a model
    # reset to `pending` is one a re-seed would happily download again, which is
    # precisely what excluding it was meant to avoid.
    logger.info(
        "`model` rows left untouched — the download endpoints already 404 on a "
        "missing blob, and a re-seed must not re-download an excluded model."
    )


if __name__ == "__main__":
    main()
