"""Blob storage client — the thin abstraction workers use for raw/processed
artifacts (server.md#object-storage).

Callers address blobs by **key** (e.g. ``raw/<uid>.glb``) and never touch
filesystem paths or GCS buckets directly, so the backend can swap without
touching worker code. The skeleton (milestone 2) uses `LocalStorage` over a
local directory; a `GcsStorage` implementing the same `Storage` protocol slots
in for cloud without changing callers.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Protocol, runtime_checkable

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


@runtime_checkable
class Storage(Protocol):
    """Key-addressed blob store. Keys are ``/``-separated (e.g. ``raw/<uid>.glb``)."""

    def exists(self, key: str) -> bool:
        """True if a blob is already stored at `key` (the idempotency check)."""
        ...

    def put_bytes(self, key: str, data: bytes) -> None:
        """Write `data` at `key`, overwriting any existing blob."""
        ...

    def get_bytes(self, key: str) -> bytes:
        """Read the blob at `key`; raises if it does not exist."""
        ...

    def list_keys(self, prefix: str) -> Iterator[str]:
        """Every key under `prefix`, in no guaranteed order.

        Yields rather than returning a list: the render prefix alone holds ~141k
        objects, and `app.reconcile_from_storage` streams the listing instead of
        materialising it. Metadata only — no blob bodies are read, so this costs
        no egress.
        """
        ...

    def signed_url(self, key: str, ttl: timedelta) -> str | None:
        """A time-limited URL a browser can fetch `key` from directly.

        Returns ``None`` when the backend can't issue one (local dev), in which
        case the caller streams the bytes through the API instead. Serving the
        browser straight from the store matters in cloud: 12 view PNGs per model
        across a paginated grid would otherwise all be proxied through the API.
        """
        ...


class LocalStorage:
    """Filesystem-backed `Storage` rooted at a directory (skeleton / local dev).

    A key maps to ``<root>/<key>``; parent directories are created on write.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    def _path(self, key: str) -> Path:
        return self._root / key

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def put_bytes(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def list_keys(self, prefix: str) -> Iterator[str]:
        """Walk the filesystem under `prefix`, yielding keys relative to the root.

        `prefix` is a key fragment, not necessarily a directory (``raw/a`` is a
        legal prefix), so this walks the nearest enclosing directory and filters —
        matching how object stores treat a prefix.
        """
        search_root = self._root / prefix
        base = search_root if search_root.is_dir() else search_root.parent
        if not base.is_dir():
            return
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            key = path.relative_to(self._root).as_posix()
            if key.startswith(prefix):
                yield key

    def signed_url(self, key: str, ttl: timedelta) -> str | None:
        """None — a local file has no URL, so the API streams the bytes itself."""
        return None


class GcsStorage:
    """GCS-backed `Storage` — keys map to objects in a bucket (cloud backend).

    The `google-cloud-storage` import is deferred to construction so LocalStorage
    users (and the test suite) don't need the dependency at import time.
    """

    def __init__(self, bucket_name: str) -> None:
        from google.cloud import storage

        self._bucket = storage.Client().bucket(bucket_name)
        self._signing_credentials = None  # lazily fetched on the first sign

    def _iam_signer(self) -> tuple[str, str]:
        """The SA email + a live access token for IAM-based V4 signing.

        Cloud Run's ambient credentials carry no private key, so signing has to go
        through the IAM ``signBlob`` API — which needs the runtime SA's own email
        and a current access token, and the runtime SA holding
        ``iam.serviceAccountTokenCreator`` on itself. The credentials are fetched
        once and the token refreshed only when stale (it lives ~1h).
        """
        import google.auth
        from google.auth.transport.requests import Request as AuthRequest

        if self._signing_credentials is None:
            credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            self._signing_credentials = credentials
        credentials = self._signing_credentials
        if not credentials.valid:
            credentials.refresh(AuthRequest())
        # Prefer the configured email so signing doesn't depend on what the
        # metadata server happens to report; fall back to the credentials' own.
        email = get_settings().signer_service_account_email or getattr(
            credentials, "service_account_email", None
        )
        return email, credentials.token

    def exists(self, key: str) -> bool:
        return self._bucket.blob(key).exists()

    def put_bytes(self, key: str, data: bytes) -> None:
        self._bucket.blob(key).upload_from_string(data)

    def get_bytes(self, key: str) -> bytes:
        return self._bucket.blob(key).download_as_bytes()

    def list_keys(self, prefix: str) -> Iterator[str]:
        """Stream object names under `prefix` (the client paginates internally)."""
        for blob in self._bucket.list_blobs(prefix=prefix):
            yield blob.name

    def signed_url(self, key: str, ttl: timedelta) -> str | None:
        """A V4 signed GET URL, so the browser reads GCS without proxying us.

        Signs through the IAM ``signBlob`` API (`service_account_email` +
        `access_token`), because Cloud Run's metadata credentials have no private
        key — without those two arguments ``generate_signed_url`` tries to sign
        locally and always raises there. This needs the runtime SA to hold
        `iam.serviceAccountTokenCreator` on itself.

        Returns None if signing fails rather than failing the request: the caller
        falls back to streaming the blob through the API, so the page still works
        (slower), and the log line points at the likely-missing IAM binding.
        """
        try:
            email, token = self._iam_signer()
            return self._bucket.blob(key).generate_signed_url(
                version="v4",
                expiration=ttl,
                method="GET",
                service_account_email=email,
                access_token=token,
            )
        except Exception:
            logger.warning(
                "could not sign a URL for %s — falling back to streaming through the API; "
                "grant the runtime service account iam.serviceAccountTokenCreator on itself",
                key,
                exc_info=True,
            )
            return None


class RoutedGcsStorage:
    """Routes keys across the two GCS buckets by prefix (server.md#object-storage).

    ``raw/*`` keys live in the raw bucket (Nearline-tiered, written once); everything
    else (``processed/*``) lives in the processed bucket (Standard, read every epoch).
    Callers still address blobs by key alone — the split is invisible to worker code,
    exactly as with the single-bucket `LocalStorage` used locally.
    """

    def __init__(self, raw_bucket: str, processed_bucket: str) -> None:
        self._raw = GcsStorage(raw_bucket)
        self._processed = GcsStorage(processed_bucket)

    def _backend(self, key: str) -> GcsStorage:
        return self._raw if key.startswith("raw/") else self._processed

    def exists(self, key: str) -> bool:
        return self._backend(key).exists(key)

    def put_bytes(self, key: str, data: bytes) -> None:
        self._backend(key).put_bytes(key, data)

    def get_bytes(self, key: str) -> bytes:
        return self._backend(key).get_bytes(key)

    def list_keys(self, prefix: str) -> Iterator[str]:
        """List from whichever bucket owns `prefix` — routing by the same rule.

        A prefix always resolves to one bucket because the split is on the leading
        ``raw/`` segment, so no listing ever needs to span both.
        """
        return self._backend(prefix).list_keys(prefix)

    def signed_url(self, key: str, ttl: timedelta) -> str | None:
        return self._backend(key).signed_url(key, ttl)


def build_storage(settings: Settings) -> Storage:
    """Return the storage backend chosen by config: LocalStorage or routed GCS."""
    if settings.storage_backend == "gcs":
        return RoutedGcsStorage(settings.raw_bucket, settings.processed_bucket)
    return LocalStorage(settings.storage_root)
