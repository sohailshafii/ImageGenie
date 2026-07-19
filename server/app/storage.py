"""Blob storage client — the thin abstraction workers use for raw/processed
artifacts (server.md#object-storage).

Callers address blobs by **key** (e.g. ``raw/<uid>.glb``) and never touch
filesystem paths or GCS buckets directly, so the backend can swap without
touching worker code. The skeleton (milestone 2) uses `LocalStorage` over a
local directory; a `GcsStorage` implementing the same `Storage` protocol slots
in for cloud without changing callers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


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
