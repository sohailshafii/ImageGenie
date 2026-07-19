from unittest.mock import MagicMock

import google.cloud.storage as gcs
import pytest

from app.storage import GcsStorage, Storage


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
