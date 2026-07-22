"""Recording and replaying failed pipeline jobs (server.md#dead-letters).

Failures are captured **by the worker at nack time**, not by reading the Pub/Sub
dead-letter queue, for three reasons:

- The error text only exists there. A dead-letter message carries the original
  payload and a delivery count — never why the subscriber gave up.
- Listing then stays a plain DB read. Pulling from a subscription to render an
  admin page would be slow and would consume the very messages it displays.
- Records outlive Pub/Sub's retention (7 days by default), so a failure from a
  month-old ingestion run is still visible.

`app/replay_dlq.py` remains the bulk tool for draining a whole DLQ back to its
topic; this module backs the per-item admin view.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from .models import DeadLetter, PipelineStage
from .queue import publish_next

# Truncated before storage: a traceback-laden message from a mesh library can run
# to kilobytes, and the admin list only needs enough to recognise the failure.
MAX_ERROR_LENGTH = 2000


def record_failure(
    session: Session,
    uid: str,
    stage: PipelineStage,
    error: str,
    delivery_attempt: int | None = None,
) -> None:
    """Upsert the failure for ``(uid, stage)``.

    An upsert, not an insert: at-least-once delivery means the same job fails
    repeatedly, and the admin wants current state rather than one row per
    attempt. Re-failing also clears ``replayed_at`` — a replayed job that failed
    again is outstanding once more.
    """
    statement = pg_insert(DeadLetter).values(
        model_uid=uid,
        stage=stage,
        error=error[:MAX_ERROR_LENGTH],
        delivery_attempt=delivery_attempt,
        failed_at=datetime.now(UTC),
        replayed_at=None,
    )
    session.execute(
        statement.on_conflict_do_update(
            constraint="uq_dead_letter_model_stage",
            set_={
                "error": statement.excluded.error,
                "delivery_attempt": statement.excluded.delivery_attempt,
                "failed_at": statement.excluded.failed_at,
                "replayed_at": None,
            },
        )
    )


def replay(session: Session, dead_letter_id: int) -> DeadLetter | None:
    """Re-enqueue a failed job on its stage topic; returns the updated row.

    The row is kept, marked with ``replayed_at``, rather than deleted — an admin
    needs to see that they already retried something. It disappears from the
    outstanding list only by succeeding, since a fresh failure resets the mark.
    """
    row = session.get(DeadLetter, dead_letter_id)
    if row is None:
        return None
    publish_next(f"{row.stage.value}-jobs", row.model_uid)
    row.replayed_at = datetime.now(UTC)
    return row


def list_dead_letters(session: Session, include_replayed: bool) -> list[DeadLetter]:
    """Failures, most recent first. Replayed ones are hidden by default."""
    query = select(DeadLetter).order_by(DeadLetter.failed_at.desc(), DeadLetter.id.desc())
    if not include_replayed:
        query = query.where(DeadLetter.replayed_at.is_(None))
    return list(session.scalars(query).all())
