"""ORM entities for the metadata DB (server.md#database).

The DB is the source of truth for pipeline state; object storage holds the heavy
blobs and the DB stores only their keys. Milestone 2 defines ``model`` (the
download stage's state); milestone 4 adds ``artifact`` (the convert / normalize /
render outputs); ``label`` / ``training_run`` / ``user`` land with their stages.
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import ARRAY, DateTime, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class DownloadStatus(str, enum.Enum):
    """Lifecycle of a model's raw-mesh download."""

    pending = "pending"
    downloaded = "downloaded"
    failed = "failed"


class Model(Base):
    """A source 3D model and its download state — one row per store object.

    ``uid`` is the store (Objaverse) id and the primary key, so the download
    worker's ``INSERT ... ON CONFLICT (uid)`` upsert is race-safe under
    at-least-once redelivery (NFR-2).
    """

    __tablename__ = "model"

    uid: Mapped[str] = mapped_column(primary_key=True)
    source_url: Mapped[str | None] = mapped_column(default=None)
    license: Mapped[str | None] = mapped_column(default=None)
    # Store metadata, backfilled by `app.backfill_metadata` (the download worker
    # doesn't persist annotations). Shown in the labeling UI to aid the decision
    # — on the ambiguous classes the title is often what settles it. Nullable
    # because a model can be ingested long before its metadata is fetched.
    title: Mapped[str | None] = mapped_column(default=None)
    tags: Mapped[list[str] | None] = mapped_column(ARRAY(String), default=None)
    download_status: Mapped[DownloadStatus] = mapped_column(default=DownloadStatus.pending)
    content_hash: Mapped[str | None] = mapped_column(default=None)
    raw_key: Mapped[str | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )


class ArtifactStage(str, enum.Enum):
    """The preprocessing stage that produced an artifact (server.md#queue)."""

    converted = "converted"
    normalized = "normalized"
    rendered = "rendered"


class ArtifactStatus(str, enum.Enum):
    """Lifecycle of a single stage's output for one model."""

    pending = "pending"
    done = "done"
    failed = "failed"


class Artifact(Base):
    """One preprocessing stage's output for one model (server.md#database).

    The DB stores only the object ``key`` (single-file stages) or key **prefix**
    (the render stage's per-view PNGs), never the blob itself. The unique
    ``(model_uid, stage)`` constraint backs the workers' ``INSERT ... ON CONFLICT``
    upsert, so at-least-once redelivery never duplicates a stage's row (NFR-2).
    """

    __tablename__ = "artifact"
    __table_args__ = (
        UniqueConstraint("model_uid", "stage", name="uq_artifact_model_stage"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_uid: Mapped[str] = mapped_column(ForeignKey("model.uid"), index=True)
    stage: Mapped[ArtifactStage] = mapped_column()
    key: Mapped[str] = mapped_column()
    content_hash: Mapped[str | None] = mapped_column(default=None)
    status: Mapped[ArtifactStatus] = mapped_column(default=ArtifactStatus.pending)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )


class PipelineStage(str, enum.Enum):
    """A processing stage a job can fail in — includes download, unlike
    ``ArtifactStage``, which only covers stages that produce an artifact."""

    download = "download"
    convert = "convert"
    normalize = "normalize"
    render = "render"


class DeadLetter(Base):
    """A job that failed a stage, with the error that caused it.

    Recorded by the worker at nack time, because **that is the only place the
    error text exists** — a Pub/Sub dead-letter message carries the original
    payload and a delivery count, never the reason. Keeping it here also means
    the admin view is a plain DB read rather than a destructive pull from the
    DLQ subscription, and that failures outlive Pub/Sub's 7-day retention.

    Unique on ``(model_uid, stage)``: a retried job that fails again updates the
    row rather than adding another, so the list shows current state and not a
    log of every attempt.
    """

    __tablename__ = "dead_letter"
    __table_args__ = (
        UniqueConstraint("model_uid", "stage", name="uq_dead_letter_model_stage"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_uid: Mapped[str] = mapped_column(index=True)  # no FK: a download can
    # fail before the model row exists, and losing the record would be worse
    stage: Mapped[PipelineStage] = mapped_column()
    error: Mapped[str] = mapped_column()
    # Pub/Sub's count for this message; at the subscription's max it stops being
    # redelivered and goes to the dead-letter topic.
    delivery_attempt: Mapped[int | None] = mapped_column(default=None)
    failed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    # Set when an admin re-enqueues it; the row is kept so the history is visible.
    replayed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )


class LabelSource(str, enum.Enum):
    """Where a label came from — the weak-labeling rules or a human correction."""

    weak = "weak"
    manual = "manual"


class Label(Base):
    """A class label for a model (server.md#database, ml.md#weak-label-policy).

    Weak (rule-derived) and manual (human-corrected via the labeling UI) labels are
    kept as **distinct rows** so weak-vs-corrected analysis stays possible — the
    frontend's "current" label for a model is its most recent one.
    """

    __tablename__ = "label"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_uid: Mapped[str] = mapped_column(ForeignKey("model.uid"), index=True)
    class_name: Mapped[str] = mapped_column()  # one of the 12-class roster (ml/taxonomy.py)
    source: Mapped[LabelSource] = mapped_column()
    confidence: Mapped[float | None] = mapped_column(default=None)  # weak labels only
    annotator: Mapped[str | None] = mapped_column(default=None)  # user email, for manual
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class UserRole(str, enum.Enum):
    """Access role — ``user`` can view, ``admin`` can also correct + upload (FR-8)."""

    user = "user"
    admin = "admin"


class User(Base):
    """An authenticated account (server.md#database, web.md#auth--roles).

    Table is ``app_user`` — ``user`` is a reserved word in PostgreSQL.
    """

    __tablename__ = "app_user"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(unique=True, index=True)
    role: Mapped[UserRole] = mapped_column(default=UserRole.user)
    password_hash: Mapped[str] = mapped_column()
    verified: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Invite(Base):
    """An admin-minted, email-bound signup invitation (web.md#auth--roles).

    Signup is invite-only, so this table is the gate on account creation. Keyed by
    ``email`` so re-inviting the same address refreshes the existing invite rather
    than accumulating rows — matching the frontend's idempotent-per-email contract.
    """

    __tablename__ = "invite"

    email: Mapped[str] = mapped_column(primary_key=True)  # normalized lowercase
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    accepted: Mapped[bool] = mapped_column(default=False)
    invited_by: Mapped[str | None] = mapped_column(default=None)  # admin's email
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class EmailVerification(Base):
    """A one-time email-verification token (web.md#auth--roles).

    Stored as a **SHA-256 hash**, never in the clear: a token grants the right to
    verify an account, so a leaked DB snapshot (or a stray log of a query result)
    shouldn't hand that out. Plain SHA-256 rather than bcrypt is right here —
    these are 256-bit random values, not guessable secrets, so there is nothing
    for a slow hash to defend against.
    """

    __tablename__ = "email_verification"

    token_hash: Mapped[str] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("app_user.id"), index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class LoginSession(Base):
    """A login session — the opaque token held in an httpOnly cookie maps here.

    Server-side sessions (not a stateless JWT) so logout can revoke immediately
    (web.md#auth--roles). ``expires_at`` is set by the app at creation. Named
    ``LoginSession`` to avoid colliding with SQLAlchemy's ``Session``.
    """

    __tablename__ = "session"

    token: Mapped[str] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("app_user.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    # timezone-aware so the app's aware expiry round-trips (naive would mismatch
    # datetime.now(timezone.utc) at comparison time).
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
