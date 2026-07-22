"""Backfill model titles and tags from Objaverse annotations.

The download worker stores the mesh but not the store metadata, so the labeling
UI has no caption beyond the uid. Titles and tags are what let a human settle the
ambiguous cases — `figure` vs `animal` sits at 0.62 weak-label precision, and the
title is very often what decides it (ml.md#weak-label-policy).

    python -m app.backfill_metadata            # every model missing metadata
    python -m app.backfill_metadata --limit 500 --dry-run

**Idempotent (NFR-2):** only rows with no title are fetched, so a rerun after a
partial pass picks up where it stopped.

⚠️ **Objaverse annotations are sharded.** `load_annotations` downloads whichever
of the ~160 shard files contain the requested uids, and our uids are scattered
across all of them — so the first run pulls most of the shard set (hundreds of MB,
cached locally afterwards). That is why this is a batch tool and not something the
API calls on demand.
"""

from __future__ import annotations

import argparse
import logging

from sqlalchemy import select

from .db import init_db, session_scope
from .models import Model

logger = logging.getLogger(__name__)

# Uids per annotation fetch. Bounds peak memory; the shard downloads dominate
# either way, and they are cached across batches.
FETCH_BATCH_SIZE = 2000


def _names(entries) -> list[str]:
    """Pull `name` out of Objaverse's ``[{"name": ...}, ...]`` shape, skipping blanks."""
    return [name for entry in entries or [] if (name := (entry or {}).get("name"))]


def extract_metadata(annotation: dict) -> tuple[str | None, list[str]]:
    """Map one Objaverse annotation to ``(title, tags)``.

    Categories are folded in alongside tags: both are free-text descriptors that
    help a labeler, and the distinction only matters to the weak-labeling rules.
    """
    title = (annotation.get("name") or "").strip() or None
    tags_set = dict.fromkeys(  # dedupe, preserving order
        _names(annotation.get("tags")) + _names(annotation.get("categories"))
    )
    return title, list(tags_set)


def backfill(limit: int | None, dry_run: bool) -> dict[str, int]:
    """Fetch annotations for models lacking a title and store them."""
    import objaverse  # deferred: heavy, and only this tool needs it

    with session_scope() as session:
        query = select(Model.uid).where(Model.title.is_(None)).order_by(Model.uid)
        if limit is not None:
            query = query.limit(limit)
        pending_uids = list(session.scalars(query).all())

    counts = {"pending": len(pending_uids), "updated": 0, "no_annotation": 0}
    if not pending_uids:
        return counts

    for start in range(0, len(pending_uids), FETCH_BATCH_SIZE):
        batch = pending_uids[start : start + FETCH_BATCH_SIZE]
        logger.info("fetching annotations for %d uids (may download shards)", len(batch))
        uid_to_annotation = objaverse.load_annotations(batch)

        with session_scope() as session:
            for uid in batch:
                annotation = uid_to_annotation.get(uid)
                if not annotation:
                    # Objaverse knows the uid (we downloaded its mesh) but the
                    # shard has no record — rare, and not worth failing the run.
                    counts["no_annotation"] += 1
                    continue
                title, tags = extract_metadata(annotation)
                if title is None and not tags:
                    counts["no_annotation"] += 1
                    continue
                if not dry_run:
                    model = session.get(Model, uid)
                    model.title = title
                    model.tags = tags
                counts["updated"] += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None, help="cap how many models to backfill"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="fetch and report, write nothing"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    init_db()
    counts = backfill(args.limit, args.dry_run)
    logger.info(
        "%d model(s) missing metadata: %d updated, %d had no usable annotation",
        counts["pending"],
        counts["updated"],
        counts["no_annotation"],
    )


if __name__ == "__main__":
    main()
