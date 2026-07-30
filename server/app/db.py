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


def _resolve_database_url(settings) -> str:
    """The connection URL, read from Secret Manager when configured.

    The Vertex training job takes this path: a job's environment is visible in
    its metadata, so the URL — which carries the DB password — is passed as a
    *secret name* and fetched at startup instead. Workers and the API leave this
    unset and keep using `database_url` directly, injected by Cloud Run.
    """
    if settings.database_url_secret is None:
        return settings.database_url

    from google.cloud import secretmanager

    client = secretmanager.SecretManagerServiceClient()
    response = client.access_secret_version(name=settings.database_url_secret)
    return response.payload.data.decode("utf-8").strip()


def _cloudsql_connector_engine(settings, url: str) -> Engine:
    """An engine that dials Cloud SQL through the Python connector.

    Used only by the Vertex training job (server.md#training-gpu). Cloud Run
    reaches Cloud SQL over a Unix socket it mounts; Vertex has no equivalent, and
    its egress IP is dynamic so the instance's authorized networks cannot cover
    it either. The connector solves both by authenticating over IAM against the
    Cloud SQL Admin API, needing no VPC and no allowlist.

    The URL's credentials and database name are reused; only the transport
    changes, so the two paths cannot drift about *which* database they mean.
    """
    from urllib.parse import unquote, urlsplit

    from google.cloud.sql.connector import Connector

    parsed = urlsplit(url)
    connector = Connector()

    def connect():
        return connector.connect(
            settings.cloudsql_instance,
            "pg8000",
            user=unquote(parsed.username or ""),
            password=unquote(parsed.password or ""),
            db=parsed.path.lstrip("/"),
        )

    # pool_pre_ping: a training run is long and mostly idle on the DB between
    # epoch writes, easily long enough for Cloud SQL to drop a stale connection.
    return create_engine("postgresql+pg8000://", creator=connect, pool_pre_ping=True, future=True)


@lru_cache
def get_engine() -> Engine:
    """Return the process-wide engine (connections are opened lazily)."""
    settings = get_settings()
    url = _resolve_database_url(settings)
    if settings.cloudsql_instance:
        return _cloudsql_connector_engine(settings, url)
    # pool_pre_ping for the same reason the connector engine above sets it, which
    # this branch was missing: a pooled connection can go stale across a long
    # compute phase that touches no SQL. `ml/evaluate.py` reads the run, spends
    # ~15 minutes scoring a split, then inserts one row — and that insert died on
    # "server closed the connection unexpectedly", losing the whole report. A
    # pre-ping costs one round trip per checkout and turns that into a reconnect.
    return create_engine(url, pool_pre_ping=True, future=True)


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
