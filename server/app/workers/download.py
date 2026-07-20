"""Download worker (FR-1) — the skeleton's one worker type.

Consumes a job ``{"uid": ...}`` from the download queue, fetches the model's mesh
via objaverse into storage, and records it in the ``model`` table.

**Idempotent (NFR-2):** a redelivered job for an already-downloaded model is
skipped, and the DB write is an ``INSERT ... ON CONFLICT (uid) DO UPDATE`` upsert,
so at-least-once delivery and concurrent redelivery never duplicate work. The
worker fills only download state; ``source_url``/``license`` are backfilled
elsewhere (fetching per-uid annotations here would hit the scattered-uid cost).
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from pathlib import Path

import objaverse
from sqlalchemy.dialects.postgresql import insert as pg_insert

from ..config import get_settings
from ..consumer import run_stage
from ..db import session_scope
from ..models import DownloadStatus, Model
from ..queue import publish_next
from ..storage import build_storage

logger = logging.getLogger(__name__)
STAGE = "download"

# Fetching from the Objaverse/HF mirror fails transiently under load — read/SSL
# timeouts, dropped connections, occasional 5xx. Retry in-process with exponential
# backoff + full jitter (server.md#request-resilience) so a transient blip doesn't
# burn a Pub/Sub delivery attempt and dead-letter a perfectly good model. Genuine
# failures (missing object, OOM) still exhaust attempts and dead-letter.
_MAX_FETCH_ATTEMPTS = 4
_BASE_BACKOFF_SECONDS = 2.0


def _raw_key(uid: str) -> str:
    return f"raw/{uid}.glb"


def _fetch_mesh(uid: str) -> bytes:
    """Download `uid`'s mesh bytes, retrying transient failures with backoff+jitter."""
    for attempt in range(1, _MAX_FETCH_ATTEMPTS + 1):
        try:
            local_path = objaverse.load_objects([uid])[uid]
            return Path(local_path).read_bytes()
        except Exception as error:  # noqa: BLE001 — mirror errors are transient; DLQ is the backstop
            if attempt == _MAX_FETCH_ATTEMPTS:
                raise
            backoff = _BASE_BACKOFF_SECONDS * 2 ** (attempt - 1)
            delay = backoff + random.uniform(0.0, backoff)  # full jitter
            logger.warning(
                "download fetch failed; retrying",
                extra={"uid": uid, "stage": STAGE, "attempt": attempt, "error": str(error)},
            )
            time.sleep(delay)
    raise RuntimeError("unreachable")  # loop either returns or raises


def process(job: dict) -> str:
    """Download one model. Returns ``"downloaded"`` or ``"skipped"``.

    Hands the model to the convert stage on either outcome, so a re-seeded or
    redelivered job for an already-downloaded model still drives the pipeline
    forward (convert is idempotent and skips if its output already exists).
    """
    uid = job["uid"]
    settings = get_settings()
    storage = build_storage(settings)
    raw_key = _raw_key(uid)

    # Idempotency check: skip if the DB says downloaded and the blob is present.
    with session_scope() as session:
        existing = session.get(Model, uid)
        already_done = (
            existing is not None
            and existing.download_status == DownloadStatus.downloaded
            and storage.exists(raw_key)
        )
    if already_done:
        logger.info("skip already-downloaded", extra={"uid": uid, "stage": STAGE})
        publish_next(settings.convert_topic, uid)
        return "skipped"

    # Fetch the mesh (retries transient mirror failures with backoff+jitter).
    data = _fetch_mesh(uid)
    content_hash = hashlib.sha256(data).hexdigest()
    storage.put_bytes(raw_key, data)

    # Idempotent upsert of the model's download state.
    with session_scope() as session:
        statement = pg_insert(Model).values(
            uid=uid,
            content_hash=content_hash,
            raw_key=raw_key,
            download_status=DownloadStatus.downloaded,
        ).on_conflict_do_update(
            index_elements=["uid"],
            set_={
                "content_hash": content_hash,
                "raw_key": raw_key,
                "download_status": DownloadStatus.downloaded,
            },
        )
        session.execute(statement)

    logger.info("downloaded", extra={"uid": uid, "stage": STAGE, "content_hash": content_hash})
    publish_next(settings.convert_topic, uid)
    return "downloaded"


def main() -> None:
    """Run the worker: bootstrap the DB/subscription, then consume jobs forever."""
    settings = get_settings()
    run_stage(settings.download_subscription, settings.download_topic, process)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
