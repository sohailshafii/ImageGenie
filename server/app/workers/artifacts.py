"""Artifact-table helpers shared by the preprocessing stages (server.md#database).

``artifact_done`` is the idempotency gate — a stage skips only when the DB row is
marked ``done`` **and** the blob is actually present, so a half-finished job (row
written, upload lost, or vice versa) re-runs. ``record_artifact`` is the race-safe
upsert keyed on ``(model_uid, stage)`` that survives at-least-once redelivery.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from ..models import Artifact, ArtifactStage, ArtifactStatus
from ..storage import Storage


def artifact_done(
    session: Session,
    uid: str,
    stage: ArtifactStage,
    storage: Storage,
    completion_key: str,
) -> bool:
    """True if `stage` for `uid` is recorded done and `completion_key` exists.

    `completion_key` is the blob whose presence proves the stage finished — the
    single output for convert/normalize, or the last per-view PNG for render.
    """
    row = session.execute(
        select(Artifact).where(
            Artifact.model_uid == uid, Artifact.stage == stage
        )
    ).scalar_one_or_none()
    return (
        row is not None
        and row.status == ArtifactStatus.done
        and storage.exists(completion_key)
    )


def record_artifact(
    session: Session,
    uid: str,
    stage: ArtifactStage,
    key: str,
    content_hash: str | None,
) -> None:
    """Upsert the ``done`` artifact row for `(uid, stage)` (idempotent, NFR-2)."""
    statement = pg_insert(Artifact).values(
        model_uid=uid,
        stage=stage,
        key=key,
        content_hash=content_hash,
        status=ArtifactStatus.done,
    ).on_conflict_do_update(
        index_elements=["model_uid", "stage"],
        set_={"key": key, "content_hash": content_hash, "status": ArtifactStatus.done},
    )
    session.execute(statement)
