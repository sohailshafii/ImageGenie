from unittest.mock import MagicMock

import google.cloud.storage as gcs
import pytest

from app.storage import GcsStorage, RoutedGcsStorage, Storage


def test_gcs_storage_conforms_and_wires_client(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = MagicMock()
    monkeypatch.setattr(gcs, "Client", lambda: fake_client)

    store = GcsStorage("imagegenie-pipeline-raw")

    assert isinstance(store, Storage)  # structural conformance to the protocol
    fake_client.bucket.assert_called_once_with("imagegenie-pipeline-raw")

    blob = fake_client.bucket.return_value.blob
    store.put_bytes("raw/abc.glb", b"MESH")
    blob.assert_called_with("raw/abc.glb")
    blob.return_value.upload_from_string.assert_called_once_with(b"MESH")


def test_routed_gcs_storage_routes_by_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    name_to_bucket: dict[str, MagicMock] = {}

    def make_bucket(name: str) -> MagicMock:
        return name_to_bucket.setdefault(name, MagicMock())

    fake_client = MagicMock()
    fake_client.bucket.side_effect = make_bucket
    monkeypatch.setattr(gcs, "Client", lambda: fake_client)

    store = RoutedGcsStorage("raw-bucket", "processed-bucket")
    store.put_bytes("raw/abc.glb", b"RAW")
    store.put_bytes("processed/converted/abc.ply", b"PROC")

    # raw/* → raw bucket; processed/* → processed bucket.
    name_to_bucket["raw-bucket"].blob.assert_called_once_with("raw/abc.glb")
    name_to_bucket["processed-bucket"].blob.assert_called_once_with("processed/converted/abc.ply")


def test_gcs_storage_lists_object_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Yields blob names, delegating pagination to the client."""
    fake_client = MagicMock()
    monkeypatch.setattr(gcs, "Client", lambda: fake_client)
    bucket = fake_client.bucket.return_value
    bucket.list_blobs.return_value = [
        MagicMock(name_attr="a") for _ in range(2)
    ]
    for blob, object_name in zip(
        bucket.list_blobs.return_value, ["raw/a.glb", "raw/b.glb"], strict=True
    ):
        blob.name = object_name

    store = GcsStorage("imagegenie-pipeline-raw")

    assert list(store.list_keys("raw/")) == ["raw/a.glb", "raw/b.glb"]
    bucket.list_blobs.assert_called_once_with(prefix="raw/")


def test_routed_gcs_storage_lists_from_the_owning_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prefix resolves to exactly one bucket, so no listing spans both."""
    name_to_bucket: dict[str, MagicMock] = {}

    def make_bucket(name: str) -> MagicMock:
        return name_to_bucket.setdefault(name, MagicMock())

    fake_client = MagicMock()
    fake_client.bucket.side_effect = make_bucket
    monkeypatch.setattr(gcs, "Client", lambda: fake_client)
    store = RoutedGcsStorage("raw-bucket", "processed-bucket")

    for bucket in name_to_bucket.values():
        bucket.list_blobs.return_value = []

    list(store.list_keys("raw/"))
    list(store.list_keys("processed/renders/"))

    name_to_bucket["raw-bucket"].list_blobs.assert_called_once_with(prefix="raw/")
    name_to_bucket["processed-bucket"].list_blobs.assert_called_once_with(
        prefix="processed/renders/"
    )
