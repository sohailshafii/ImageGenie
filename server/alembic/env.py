"""Alembic environment (server.md#migrations).

Two deliberate departures from the generated template:

- **The URL comes from app config, not `alembic.ini`.** The connection string
  carries the DB password and in cloud is injected from Secret Manager, so it
  must never sit in a tracked file.
- **`target_metadata` is the app's `Base.metadata`**, so `--autogenerate` diffs
  the real models against the live schema.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app import models  # noqa: F401 — registers every table on Base.metadata
from app.config import get_settings
from app.db import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Only fall back to app config when the caller hasn't supplied a URL. Overriding
# unconditionally would silently migrate the *default* database when a caller —
# a test, or an operator pointing at one environment — asked for another.
if not config.get_main_option("sqlalchemy.url", None):
    config.set_main_option("sqlalchemy.url", get_settings().database_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting — for reviewing a migration."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Without this, autogenerate silently ignores column type changes.
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
