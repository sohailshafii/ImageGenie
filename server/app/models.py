"""ORM entities for the metadata DB (server.md#database).

The DB is the source of truth for pipeline state; object storage holds the heavy
blobs and the DB stores only their keys. Milestone 2 defines ``model`` (the
download stage's state); milestone 4 adds ``artifact`` (the convert / normalize /
render outputs); ``label`` / ``training_run`` / ``user`` land with their stages.
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import ForeignKey, UniqueConstraint, func
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
