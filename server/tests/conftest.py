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
    api.signup_limiter.reset()
    api.login_backoff.reset()


@pytest.fixture
def mail_outbox(monkeypatch: pytest.MonkeyPatch) -> Iterator[list]:
    """Capture outgoing email instead of sending it, and yield the messages.

    Configures an API key so the builder actually runs — without one the app
    short-circuits to logging the link, which would leave subjects, bodies, and
    the generated links untested.
    """
    from app import config, mail

    monkeypatch.setenv("IMAGEGENIE_RESEND_API_KEY", "test-key")
    monkeypatch.setenv("IMAGEGENIE_APP_BASE_URL", "https://app.test")
    config.get_settings.cache_clear()  # settings are memoized per process

    outbox: list[mail.OutgoingEmail] = []
    mail.set_mail_sender(lambda email, _from, _key: outbox.append(email))
    yield outbox
    mail.reset_mail_sender()
    config.get_settings.cache_clear()


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
