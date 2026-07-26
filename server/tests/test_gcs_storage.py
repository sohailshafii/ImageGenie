import os
import pickle
from unittest.mock import MagicMock

import google.cloud.storage as gcs
import pytest

from app.storage import GcsStorage, RoutedGcsStorage, Storage


def test_gcs_storage_conforms_and_wires_client(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = MagicMock()
    monkeypatch.setattr(gcs, "Client", lambda: fake_client)

    store = GcsStorage("imagegenie-pipeline-raw")

    assert isinstance(store, Storage)  # structural conformance to the protocol
    # Construction builds nothing: the client belongs to whichever process first
    # uses it, which is what lets this object cross a process boundary.
    fake_client.bucket.assert_not_called()

    blob = fake_client.bucket.return_value.blob
    store.put_bytes("raw/abc.glb", b"MESH")
    fake_client.bucket.assert_called_with("imagegenie-pipeline-raw")
    blob.assert_called_with("raw/abc.glb")
    blob.return_value.upload_from_string.assert_called_once_with(b"MESH")


# ── Crossing a process boundary (ml/dataset.py + DataLoader num_workers>0) ──
# A `storage.Client` cannot travel by either route: spawn pickles the Dataset and
# the client refuses, while fork hands the child the parent's HTTPS connection
# pool. Only the bucket name travels; the client is rebuilt per process.


def test_pickles_by_name_and_rebuilds_the_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """The spawn case: a real `storage.Client` raises on pickle, so the state
    that crosses must not contain one."""
    fake_client = MagicMock()
    monkeypatch.setattr(gcs, "Client", lambda: fake_client)
    store = GcsStorage("imagegenie-pipeline-processed")
    store.get_bytes("processed/x.ply")  # force a client into existence

    revived = pickle.loads(pickle.dumps(store))

    assert revived._bucket_name == "imagegenie-pipeline-processed"
    assert revived._client is None  # nothing carried over
    revived.get_bytes("processed/x.ply")
    fake_client.bucket.assert_called_with("imagegenie-pipeline-processed")


def test_a_forked_child_does_not_reuse_the_inherited_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fork case: no pickling happens, so the guard has to be the PID. A
    child that kept its parent's client would share an HTTPS connection pool
    across processes."""
    clients = [MagicMock(), MagicMock()]
    monkeypatch.setattr(gcs, "Client", lambda: clients.pop(0))
    monkeypatch.setattr(os, "getpid", lambda: 1000)
    store = GcsStorage("bucket")
    store.get_bytes("processed/x.ply")
    parent_client = store._client

    monkeypatch.setattr(os, "getpid", lambda: 1001)  # "fork"
    store.get_bytes("processed/x.ply")

    assert store._client is not parent_client
    assert store._client_pid == 1001


def test_the_same_process_reuses_its_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rebuilding per call would mean a fresh connection pool for every one of
    the ~141k reads an epoch makes."""
    built = []

    def make_client() -> MagicMock:
        built.append(MagicMock())
        return built[-1]

    monkeypatch.setattr(gcs, "Client", make_client)
    store = GcsStorage("bucket")

    for _ in range(3):
        store.get_bytes("processed/x.ply")

    assert len(built) == 1


def test_routed_storage_survives_the_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """The routed wrapper is what cloud actually builds, so it is what a
    DataLoader worker would receive."""
    monkeypatch.setattr(gcs, "Client", lambda: MagicMock())

    revived = pickle.loads(pickle.dumps(RoutedGcsStorage("raw-bucket", "processed-bucket")))

    assert revived._raw._bucket_name == "raw-bucket"
    assert revived._processed._bucket_name == "processed-bucket"


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


def _signable_store(monkeypatch: pytest.MonkeyPatch, *, valid: bool = True) -> tuple:
    """A GcsStorage whose default credentials are stubbed, plus the fake creds."""
    fake_client = MagicMock()
    monkeypatch.setattr(gcs, "Client", lambda: fake_client)

    creds = MagicMock()
    creds.valid = valid
    creds.token = "ya29.fake-access-token"
    creds.service_account_email = "runtime@imagegenie-pipeline.iam.gserviceaccount.com"
    import google.auth

    monkeypatch.setattr(google.auth, "default", lambda scopes=None: (creds, "proj"))
    return GcsStorage("imagegenie-pipeline-processed"), fake_client, creds


def test_signed_url_uses_the_iam_signblob_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fix: pass service_account_email + access_token, or Cloud Run can't sign."""
    from datetime import timedelta

    store, fake_client, _ = _signable_store(monkeypatch)
    blob = fake_client.bucket.return_value.blob
    blob.return_value.generate_signed_url.return_value = "https://signed.example/x"

    url = store.signed_url("processed/renders/x/view_00.png", timedelta(minutes=15))

    assert url == "https://signed.example/x"
    _, kwargs = blob.return_value.generate_signed_url.call_args
    assert kwargs["version"] == "v4"
    assert kwargs["method"] == "GET"
    assert kwargs["access_token"] == "ya29.fake-access-token"
    assert kwargs["service_account_email"].startswith("runtime@")


def test_signed_url_prefers_the_configured_signer_email(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config wins over the credentials' reported email, so signing is deterministic."""
    from datetime import timedelta

    from app import config, storage

    explicit_email = "explicit@proj.iam.gserviceaccount.com"
    monkeypatch.setattr(
        storage,
        "get_settings",
        lambda: config.Settings(signer_service_account_email=explicit_email),
    )
    store, fake_client, _ = _signable_store(monkeypatch)
    blob = fake_client.bucket.return_value.blob

    store.signed_url("processed/x.ply", timedelta(minutes=15))

    _, kwargs = blob.return_value.generate_signed_url.call_args
    assert kwargs["service_account_email"] == "explicit@proj.iam.gserviceaccount.com"


def test_signed_url_refreshes_a_stale_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """A token past its ~1h life is refreshed before it's used to sign."""
    from datetime import timedelta

    store, fake_client, creds = _signable_store(monkeypatch, valid=False)

    store.signed_url("processed/x.ply", timedelta(minutes=15))

    creds.refresh.assert_called_once()


def test_signed_url_returns_none_when_signing_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A signing failure degrades to streaming (None), not a 500."""
    from datetime import timedelta

    store, fake_client, _ = _signable_store(monkeypatch)
    blob = fake_client.bucket.return_value.blob
    blob.return_value.generate_signed_url.side_effect = RuntimeError("no signBlob permission")

    assert store.signed_url("processed/x.ply", timedelta(minutes=15)) is None
