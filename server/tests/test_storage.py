from pathlib import Path

from app.storage import LocalStorage, Storage


def test_local_storage_round_trip(tmp_path: Path) -> None:
    store = LocalStorage(tmp_path)
    key = "raw/abc123.glb"

    assert isinstance(store, Storage)  # structural conformance to the protocol
    assert not store.exists(key)

    store.put_bytes(key, b"mesh-bytes")

    assert store.exists(key)
    assert store.get_bytes(key) == b"mesh-bytes"
    assert (tmp_path / "raw" / "abc123.glb").is_file()  # nested key created dirs
