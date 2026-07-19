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
