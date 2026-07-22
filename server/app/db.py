"""SQLAlchemy engine, session, and declarative base for the metadata DB.

Postgres everywhere — dev (Docker Compose) and prod (Cloud SQL) share one dialect
so the idempotent upserts (INSERT ... ON CONFLICT) and row-level concurrency the
workers rely on behave identically (server.md#database).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session

from .config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Declarative base shared by all ORM models."""


@lru_cache
def get_engine() -> Engine:
    """Return the process-wide engine (connections are opened lazily)."""
    return create_engine(get_settings().database_url, future=True)


def init_db() -> None:
    """Materialize the schema at startup — **local convenience only**.

    Alembic owns the schema (server.md#migrations). This stays because a fresh
    local Postgres should just work when a worker starts, but it is off unless
    ``IMAGEGENIE_AUTO_CREATE_SCHEMA`` says otherwise, for two reasons:

    - ``create_all`` and Alembic must not both create tables. If ``create_all``
      wins the race, the migration that would have created that table fails with
      "already exists" — and the version table then disagrees with reality.
    - It silently does the wrong thing on a schema *change*: it adds missing
      tables but never alters an existing one, so a new column just never appears
      and the failure surfaces later, as a query error.

    Deployed environments run ``alembic upgrade head`` as a deploy step instead —
    once, rather than racing from every worker instance.
    """
    from . import models  # noqa: F401 — import registers the tables on Base.metadata

    if not get_settings().auto_create_schema:
        logger.debug("auto_create_schema off — schema is managed by Alembic")
        return
    logger.warning(
        "creating schema with create_all (IMAGEGENIE_AUTO_CREATE_SCHEMA) — local "
        "convenience only; deployed environments must run 'alembic upgrade head'"
    )
    Base.metadata.create_all(get_engine())


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session scope: commit on success, roll back on error."""
    session = Session(get_engine())
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
