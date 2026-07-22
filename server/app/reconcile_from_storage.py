"""Rebuild the `model` and `artifact` tables from what is in object storage.

Object storage is the durable record; the tables are an index over it. Every key
carries its model uid (`artifact_keys`), so a bucket listing is enough to
reconstruct both tables without re-downloading or re-rendering anything. That is
what makes "drop the schema and `alembic upgrade head`" a cheap option for
adopting migrations on the existing Cloud SQL database (server.md#migrations),
rather than one that costs a fresh ingestion run.

    python -m app.reconcile_from_storage --dry-run     # report, change nothing
    python -m app.reconcile_from_storage

Against cloud, point it at the real buckets and database:

    IMAGEGENIE_STORAGE_BACKEND=gcs IMAGEGENIE_DATABASE_URL=... \
        python -m app.reconcile_from_storage

**Idempotent (NFR-2).** Rows are upserted, so a rerun over the same buckets is a
no-op and an interrupted run can simply be restarted.

**Listing only — no blob bodies are read**, so this costs no egress even over
~165k objects.

Two things it deliberately does *not* restore:

- **`content_hash`.** The workers store a sha256 of the bytes; the object store
  keeps its own md5/crc32c, which is a different digest. Recovering it would mean
  downloading every blob — real egress against NFR-1's budget — and nothing reads
  the hash for correctness: the stage-skip gate checks the row status and the
  blob's presence, not its digest. Existing hashes are preserved, never
  overwritten with null.
- **`title` / `tags`.** These come from the store's annotations, not from the
  blobs. Run `app.backfill_metadata` afterwards; this tool leaves any already
  present untouched.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from collections.abc import Iterator

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from .artifact_keys import (
    CONVERTED_PREFIX,
    NORMALIZED_PREFIX,
    NUM_VIEWS,
    RAW_PREFIX,
    RENDERS_PREFIX,
    converted_key,
    normalized_key,
    raw_key,
    renders_prefix,
    uid_from_key,
)
from .db import init_db, session_scope
from .models import Artifact, ArtifactStage, ArtifactStatus, DownloadStatus, Model
from .storage import Storage, build_storage

logger = logging.getLogger(__name__)

# Rows per INSERT. Large enough that 12k models is a handful of round trips,
# small enough not to build a giant statement in memory.
INSERT_BATCH_SIZE = 1000


def _uids_under(storage: Storage, prefix: str) -> tuple[set[str], int]:
    """Every uid appearing under `prefix`, plus a count of unrecognised keys."""
    uids_set: set[str] = set()
    unrecognised = 0
    for key in storage.list_keys(prefix):
        uid = uid_from_key(key)
        if uid is None:
            unrecognised += 1
            continue
        uids_set.add(uid)
    return uids_set, unrecognised


def _complete_render_uids(storage: Storage) -> tuple[set[str], set[str]]:
    """Split render uids into (complete, partial) by whether all views are present.

    A model whose render stage died midway leaves some of its `NUM_VIEWS` PNGs
    behind. Recording that as `done` would permanently hide it from a rerun, since
    the stage-skip gate trusts the row — so partial sets are excluded here and
    reported instead, leaving the model eligible for re-rendering.
    """
    uid_to_view_count: Counter[str] = Counter()
    for key in storage.list_keys(RENDERS_PREFIX):
        uid = uid_from_key(key)
        if uid is not None:
            uid_to_view_count[uid] += 1

    complete_set = {uid for uid, count in uid_to_view_count.items() if count >= NUM_VIEWS}
    partial_set = set(uid_to_view_count) - complete_set
    return complete_set, partial_set


def _batched(rows: list[dict], size: int) -> Iterator[list[dict]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _upsert_models(session: Session, uids_with_raw_set: set[str], all_uids_set: set[str]) -> None:
    """Insert one `model` row per uid, preserving any metadata already present.

    The conflict clause updates only what storage is authoritative for. `title`
    and `tags` are left alone so a rerun doesn't undo `app.backfill_metadata`.
    """
    rows = [
        {
            "uid": uid,
            "raw_key": raw_key(uid) if uid in uids_with_raw_set else None,
            "download_status": (
                DownloadStatus.downloaded if uid in uids_with_raw_set else DownloadStatus.pending
            ),
        }
        for uid in sorted(all_uids_set)
    ]
    for batch in _batched(rows, INSERT_BATCH_SIZE):
        statement = pg_insert(Model).values(batch)
        session.execute(
            statement.on_conflict_do_update(
                index_elements=["uid"],
                set_={
                    "raw_key": statement.excluded.raw_key,
                    "download_status": statement.excluded.download_status,
                },
            )
        )


def _upsert_artifacts(session: Session, stage: ArtifactStage, uid_to_key: dict[str, str]) -> None:
    """Insert the `done` artifact rows for one stage, keyed on (model_uid, stage).

    `content_hash` is coalesced rather than assigned: the reconciler has no hash
    to offer, and overwriting a real one with null would lose information a
    previous ingestion recorded.
    """
    rows = [
        {
            "model_uid": uid,
            "stage": stage,
            "key": key,
            "status": ArtifactStatus.done,
            "content_hash": None,
        }
        for uid, key in sorted(uid_to_key.items())
    ]
    for batch in _batched(rows, INSERT_BATCH_SIZE):
        statement = pg_insert(Artifact).values(batch)
        session.execute(
            statement.on_conflict_do_update(
                index_elements=["model_uid", "stage"],
                set_={
                    "key": statement.excluded.key,
                    "status": statement.excluded.status,
                    # The bare column refers to the *existing* row, so this keeps a
                    # hash a real ingestion recorded and only fills a null one.
                    "content_hash": func.coalesce(
                        Artifact.content_hash, statement.excluded.content_hash
                    ),
                },
            )
        )


def reconcile(storage: Storage, dry_run: bool) -> dict[str, int]:
    """Rebuild both tables from `storage`; return a count per category."""
    raw_uids_set, stray_raw = _uids_under(storage, RAW_PREFIX)
    converted_uids_set, stray_converted = _uids_under(storage, CONVERTED_PREFIX)
    normalized_uids_set, stray_normalized = _uids_under(storage, NORMALIZED_PREFIX)
    rendered_uids_set, partial_renders_set = _complete_render_uids(storage)

    all_uids_set = raw_uids_set | converted_uids_set | normalized_uids_set | rendered_uids_set
    # A model whose raw mesh is gone but whose outputs survive: still worth a row,
    # since the labeling UI reads the processed artifacts, not the source mesh.
    processed_without_raw_set = all_uids_set - raw_uids_set

    counts = {
        "models": len(all_uids_set),
        "raw": len(raw_uids_set),
        "converted": len(converted_uids_set),
        "normalized": len(normalized_uids_set),
        "rendered": len(rendered_uids_set),
        "partial_renders_skipped": len(partial_renders_set),
        "processed_without_raw": len(processed_without_raw_set),
        "unrecognised_keys": stray_raw + stray_converted + stray_normalized,
    }

    if dry_run:
        logger.info("dry run — no rows written")
        return counts

    with session_scope() as session:
        # Models first: `artifact.model_uid` is a foreign key, and the flush makes
        # the parent rows visible to the child inserts in the same transaction.
        _upsert_models(session, raw_uids_set, all_uids_set)
        session.flush()

        for stage, uid_to_key in (
            (
                ArtifactStage.converted,
                {uid: converted_key(uid) for uid in converted_uids_set},
            ),
            (
                ArtifactStage.normalized,
                {uid: normalized_key(uid) for uid in normalized_uids_set},
            ),
            (
                ArtifactStage.rendered,
                {uid: renders_prefix(uid) for uid in rendered_uids_set},
            ),
        ):
            _upsert_artifacts(session, stage, uid_to_key)

    return counts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="scan and report without writing any rows",
    )
    args = parser.parse_args()

    from .config import get_settings

    if not args.dry_run:
        init_db()
    counts = reconcile(build_storage(get_settings()), dry_run=args.dry_run)

    for name, value in counts.items():
        logger.info("%-24s %d", name, value)
    if counts["partial_renders_skipped"]:
        logger.warning(
            "%d models have an incomplete render set and were left unrecorded — "
            "republish them to the render stage to finish",
            counts["partial_renders_skipped"],
        )


if __name__ == "__main__":
    main()
