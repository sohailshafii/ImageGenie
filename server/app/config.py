"""Backend configuration from environment variables.

Per the server coding standard (config via env vars / secrets, never hardcoded),
every setting is read from an ``IMAGEGENIE_``-prefixed variable — e.g.
``IMAGEGENIE_DATABASE_URL``. Defaults target the local Docker Compose skeleton;
cloud overrides them via the environment.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="IMAGEGENIE_", extra="ignore")

    # Postgres — Docker Compose locally, Cloud SQL in prod (server.md#database).
    database_url: str = (
        "postgresql+psycopg://imagegenie:imagegenie@localhost:5432/imagegenie"
    )
    # Root directory for LocalStorage blobs in the skeleton (server.md#object-storage).
    storage_root: Path = Path("data/storage")


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings (read from the environment once)."""
    return Settings()
