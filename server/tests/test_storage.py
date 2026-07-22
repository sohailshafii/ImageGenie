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


def test_local_storage_lists_keys_under_a_prefix(tmp_path: Path) -> None:
    """Listing returns keys relative to the root, filtered by prefix."""
    store = LocalStorage(tmp_path)
    for key in (
        "raw/one.glb",
        "raw/two.glb",
        "processed/converted/one.ply",
    ):
        store.put_bytes(key, b"x")

    assert set(store.list_keys("raw/")) == {"raw/one.glb", "raw/two.glb"}
    assert set(store.list_keys("processed/")) == {"processed/converted/one.ply"}


def test_local_storage_lists_recursively_and_handles_partial_prefixes(
    tmp_path: Path,
) -> None:
    """A prefix is a key fragment, not necessarily a directory — as in an object store."""
    store = LocalStorage(tmp_path)
    store.put_bytes("processed/renders/abc/view_00.png", b"x")
    store.put_bytes("processed/renders/abd/view_00.png", b"x")

    # Recurses into per-model directories.
    assert len(set(store.list_keys("processed/renders/"))) == 2
    # Mid-segment prefix selects one model without matching its sibling.
    assert set(store.list_keys("processed/renders/abc")) == {
        "processed/renders/abc/view_00.png"
    }


def test_local_storage_lists_nothing_for_a_missing_prefix(tmp_path: Path) -> None:
    """An empty listing, not an error — the reconciler scans prefixes that may be unused."""
    assert list(LocalStorage(tmp_path).list_keys("processed/renders/")) == []
