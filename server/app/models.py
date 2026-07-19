"""ORM entities for the metadata DB (server.md#database).

The DB is the source of truth for pipeline state; object storage holds the heavy
blobs and the DB stores only their keys. Milestone 2 defines ``model`` (the
download stage's state); ``artifact`` / ``label`` / ``training_run`` / ``user``
land with the stages that use them.
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import func
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
