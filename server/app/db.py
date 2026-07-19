"""SQLAlchemy engine, session, and declarative base for the metadata DB.

Postgres everywhere — dev (Docker Compose) and prod (Cloud SQL) share one dialect
so the idempotent upserts (INSERT ... ON CONFLICT) and row-level concurrency the
workers rely on behave identically (server.md#database).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session

from .config import get_settings


class Base(DeclarativeBase):
    """Declarative base shared by all ORM models."""


@lru_cache
def get_engine() -> Engine:
    """Return the process-wide engine (connections are opened lazily)."""
    return create_engine(get_settings().database_url, future=True)


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
