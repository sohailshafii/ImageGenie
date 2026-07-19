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
from pathlib import Path

import objaverse
from google.cloud import pubsub_v1
from sqlalchemy.dialects.postgresql import insert as pg_insert

from ..config import get_settings
from ..consumer import consume
from ..db import init_db, session_scope
from ..models import DownloadStatus, Model
from ..queue import ensure_subscription
from ..storage import LocalStorage

logger = logging.getLogger(__name__)
STAGE = "download"


def _raw_key(uid: str) -> str:
    return f"raw/{uid}.glb"


def process(job: dict) -> str:
    """Download one model. Returns ``"downloaded"`` or ``"skipped"``."""
    uid = job["uid"]
    storage = LocalStorage(get_settings().storage_root)
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
        return "skipped"

    # Fetch the mesh — objaverse downloads to its cache and returns the local path.
    local_path = objaverse.load_objects([uid])[uid]
    data = Path(local_path).read_bytes()
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
    return "downloaded"


def main() -> None:
    """Run the worker: bootstrap the DB/subscription, then consume jobs forever."""
    init_db()
    settings = get_settings()
    publisher = pubsub_v1.PublisherClient()
    subscriber = pubsub_v1.SubscriberClient()
    ensure_subscription(
        subscriber, publisher, settings.download_subscription, settings.download_topic
    )
    logger.info("download worker consuming %s", settings.download_subscription)
    consume(settings.download_subscription, process)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
