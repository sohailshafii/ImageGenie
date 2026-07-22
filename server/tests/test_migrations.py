"""Schema migrations (server.md#migrations).

The load-bearing test is the last one: **the migrations must produce exactly the
schema the models describe.** If they drift, every environment built from
migrations differs from the one the tests run against — and nothing else would
notice, because the test suite builds its schema with `create_all`.
"""

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import Engine, create_engine, text
from testcontainers.postgres import PostgresContainer

from app import db
from app.db import Base


def _alembic_config(database_url: str) -> Config:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    return config


@pytest.fixture(scope="module")
def migrated_engine():
    """A Postgres built **only** by migrations — never by create_all."""
    with PostgresContainer("postgres:16-alpine", driver="psycopg") as postgres:
        url = postgres.get_connection_url()
        command.upgrade(_alembic_config(url), "head")
        engine = create_engine(url, future=True)
        yield engine
        engine.dispose()


def test_upgrade_creates_every_table(migrated_engine: Engine) -> None:
    with migrated_engine.connect() as connection:
        tables = set(
            connection.scalars(
                text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            ).all()
        )
    assert {
        "model",
        "artifact",
        "label",
        "app_user",
        "session",
        "invite",
        "email_verification",
    } <= tables


def test_migrations_match_the_models(migrated_engine: Engine) -> None:
    """No drift between `alembic upgrade head` and `Base.metadata`.

    This is what stops a model change from shipping without a migration: the
    suite's own schema comes from create_all, so nothing else would catch it.
    """
    with migrated_engine.connect() as connection:
        context = MigrationContext.configure(connection, opts={"compare_type": True})
        differences = compare_metadata(context, Base.metadata)

    assert differences == [], (
        "models and migrations disagree — run `make migration MSG=...` "
        f"to generate the missing revision. Drift: {differences}"
    )


def test_auto_create_schema_is_off_by_default() -> None:
    """create_all must not race Alembic in a deployed environment; a fresh
    Settings has to default to off, whatever the local shell has exported."""
    from app.config import Settings

    assert Settings(_env_file=None).auto_create_schema is False


def test_init_db_does_nothing_when_the_flag_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    created = []
    monkeypatch.setattr(
        Base.metadata, "create_all", lambda *args, **kwargs: created.append(args)
    )
    monkeypatch.setenv("IMAGEGENIE_AUTO_CREATE_SCHEMA", "false")
    from app import config

    config.get_settings.cache_clear()
    try:
        db.init_db()
        assert created == []
    finally:
        config.get_settings.cache_clear()
