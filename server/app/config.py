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
    # Let startup create the schema with create_all instead of migrations
    # (server.md#migrations). **Local convenience only** — Alembic owns the schema,
    # and the two must not both create tables. Deployed environments leave this
    # off and run `alembic upgrade head` as a deploy step.
    auto_create_schema: bool = False

    # Blob storage: "local" (LocalStorage over storage_root, skeleton) or "gcs"
    # (routes raw/* → raw_bucket, processed/* → processed_bucket, cloud). See
    # server.md#object-storage.
    storage_backend: str = "local"
    storage_root: Path = Path("data/storage")
    raw_bucket: str = "imagegenie-pipeline-raw"
    processed_bucket: str = "imagegenie-pipeline-processed"

    # Auth cookies (server.md#api-layer). `Secure` is off by default so local dev
    # over plain http works; every deployed environment must set
    # IMAGEGENIE_COOKIE_SECURE=true so the session cookie never crosses the wire
    # in cleartext.
    cookie_secure: bool = False

    # Whether to believe X-Forwarded-For when keying per-IP rate limits
    # (server.md#rate-limiting). Off by default: trusting the header when the app
    # is NOT behind a proxy lets a caller spoof an IP per request and walk around
    # every per-IP cap. Turn on only when a trusted proxy (Cloud Run's front end)
    # sets it.
    trust_proxy_headers: bool = False

    # Transactional email via Resend (server.md#email). The key is OPTIONAL: with
    # it unset the app logs verification/invite links instead of sending them, so
    # local dev needs no credentials — but the link then lands in the logs, so
    # every deployed environment must set IMAGEGENIE_RESEND_API_KEY.
    resend_api_key: str | None = None
    # Resend's sandbox sender delivers only to the Resend account owner's own
    # address. Real delivery needs a verified domain with SPF/DKIM.
    mail_from: str = "onboarding@resend.dev"
    # Origin used to build links in emails — the frontend, not the API.
    app_base_url: str = "http://localhost:5173"

    # Which stage handler this push service runs (server.md#compute) — the deployed
    # Cloud Run service sets IMAGEGENIE_STAGE; web.py dispatches on it.
    stage: str = "download"

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
