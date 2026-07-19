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
