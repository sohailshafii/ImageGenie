from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, create_engine
from testcontainers.postgres import PostgresContainer

from app import (
    db,
    models,  # noqa: F401  — registers tables on Base.metadata
)


@pytest.fixture(autouse=True)
def reset_rate_limiters() -> None:
    """Clear the API's module-level limiters so windows don't leak across tests.

    Without this, the per-IP login cap counts every test's logins against the
    shared TestClient address and later tests start failing with 429s.
    """
    from app import api

    api.login_limiter.reset()
    api.label_limiter.reset()
    api.login_backoff.reset()


@pytest.fixture(scope="session")
def pg_engine() -> Iterator[Engine]:
    """A real Postgres (via testcontainers) with the schema created.

    Real Postgres — not SQLite — so the workers' INSERT ... ON CONFLICT upserts
    and row-level concurrency are exercised as in prod (server.md#database).
    """
    with PostgresContainer("postgres:16-alpine", driver="psycopg") as postgres:
        engine = create_engine(postgres.get_connection_url(), future=True)
        db.Base.metadata.create_all(engine)
        yield engine
        engine.dispose()
