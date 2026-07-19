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
    # Blob storage: "local" (LocalStorage over storage_root, skeleton) or "gcs"
    # (GcsStorage over raw_bucket, cloud). See server.md#object-storage.
    storage_backend: str = "local"
    storage_root: Path = Path("data/storage")
    raw_bucket: str = "imagegenie-pipeline-raw"

    # Pub/Sub — emulator locally (set PUBSUB_EMULATOR_HOST), managed in prod
    # (server.md#queue). One topic + subscription per stage boundary; each stage
    # publishes the next stage's job (download → convert → normalize → render).
    pubsub_project: str = "imagegenie-local"
    download_topic: str = "download-jobs"
    download_subscription: str = "download-worker"
    convert_topic: str = "convert-jobs"
    convert_subscription: str = "convert-worker"
    normalize_topic: str = "normalize-jobs"
    normalize_subscription: str = "normalize-worker"
    render_topic: str = "render-jobs"
    render_subscription: str = "render-worker"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings (read from the environment once)."""
    return Settings()
