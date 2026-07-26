"""Downloading a model's meshes (server.md#downloads).

The interesting cases are the *absent* ones. A model part-way through the
pipeline has no normalized mesh, and one whose raw blob was never recorded has
no source mesh — both must read as "not available", not as a server error, and
neither may leak a soft-deleted model.
"""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text

from app import api, db
from app.artifact_keys import normalized_key, raw_key
from app.models import DownloadStatus, Model, User, UserRole
from app.security import CSRF_COOKIE, CSRF_HEADER, hash_password

VIEWER_EMAIL = "viewer@imagegenie.dev"
PASSWORD = "genie-viewer"


class FakeStorage:
    """In-memory Storage — downloads never sign, so signing is irrelevant here."""

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}

    def exists(self, key: str) -> bool:
        return key in self.blobs

    def put_bytes(self, key: str, data: bytes) -> None:
        self.blobs[key] = data

    def get_bytes(self, key: str) -> bytes:
        return self.blobs[key]

    def signed_url(self, key: str, ttl: timedelta) -> str | None:
        return None


@pytest.fixture
def storage(monkeypatch: pytest.MonkeyPatch) -> FakeStorage:
    fake = FakeStorage()
    monkeypatch.setattr(api, "build_storage", lambda _settings: fake)
    return fake


@pytest.fixture
def client(pg_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A **viewer** session — downloads are login-gated, not admin-gated."""
    monkeypatch.setattr(db, "get_engine", lambda: pg_engine)
    with pg_engine.begin() as connection:
        connection.execute(
            text(
                "TRUNCATE email_verification, invite, session, label, artifact, model,"
                " app_user RESTART IDENTITY CASCADE"
            )
        )
    with db.session_scope() as session:
        session.add(
            User(
                email=VIEWER_EMAIL,
                role=UserRole.user,
                password_hash=hash_password(PASSWORD),
                verified=True,
            )
        )
        session.add(
            Model(
                uid="glb-model",
                download_status=DownloadStatus.downloaded,
                raw_key=raw_key("glb-model"),
            )
        )
        # An admin upload — the source format is not always GLB.
        session.add(
            Model(
                uid="stl-model",
                download_status=DownloadStatus.downloaded,
                raw_key=raw_key("stl-model", ".stl"),
            )
        )
        # Queued but never fetched: no raw blob, so no key.
        session.add(Model(uid="no-raw", download_status=DownloadStatus.pending))
    test_client = TestClient(api.app)
    test_client.post("/auth/login", json={"email": VIEWER_EMAIL, "password": PASSWORD})
    test_client.headers[CSRF_HEADER] = test_client.cookies[CSRF_COOKIE]
    return test_client


def test_source_mesh_downloads_with_its_own_extension(
    client: TestClient, storage: FakeStorage
) -> None:
    storage.put_bytes(raw_key("glb-model"), b"glb-bytes")

    response = client.get("/models/glb-model/download/source")

    assert response.status_code == 200
    assert response.content == b"glb-bytes"
    assert response.headers["content-disposition"] == 'attachment; filename="glb-model.glb"'


def test_source_mesh_keeps_a_non_glb_upload_format(
    client: TestClient, storage: FakeStorage
) -> None:
    """The suffix comes off the stored key. Handing an STL back named `.glb`
    would produce a file nothing can open."""
    storage.put_bytes(raw_key("stl-model", ".stl"), b"stl-bytes")

    response = client.get("/models/stl-model/download/source")

    assert response.headers["content-disposition"] == 'attachment; filename="stl-model.stl"'


def test_normalized_mesh_downloads(client: TestClient, storage: FakeStorage) -> None:
    storage.put_bytes(normalized_key("glb-model"), b"ply-bytes")

    response = client.get("/models/glb-model/download/normalized")

    assert response.status_code == 200
    assert response.content == b"ply-bytes"
    assert (
        response.headers["content-disposition"]
        == 'attachment; filename="glb-model-normalized.ply"'
    )


def test_a_model_with_no_raw_key_is_404(client: TestClient, storage: FakeStorage) -> None:
    assert client.get("/models/no-raw/download/source").status_code == 404


def test_a_recorded_key_whose_blob_is_gone_is_404(
    client: TestClient, storage: FakeStorage
) -> None:
    """The row says the blob exists; storage disagrees. Absent, not broken."""
    assert client.get("/models/glb-model/download/source").status_code == 404


def test_an_unnormalized_model_is_404(client: TestClient, storage: FakeStorage) -> None:
    """Normalize hasn't run yet — a legitimate state, not an error."""
    assert client.get("/models/glb-model/download/normalized").status_code == 404


def test_unknown_model_is_404(client: TestClient, storage: FakeStorage) -> None:
    assert client.get("/models/nope/download/source").status_code == 404
    assert client.get("/models/nope/download/normalized").status_code == 404


def test_a_soft_deleted_model_cannot_be_downloaded(
    client: TestClient, storage: FakeStorage
) -> None:
    """Deleted means gone from the UI's point of view; the blobs stay, but the
    download must not be a way around that."""
    storage.put_bytes(raw_key("glb-model"), b"glb-bytes")
    with db.session_scope() as session:
        session.get(Model, "glb-model").deleted_at = datetime.now(UTC)

    assert client.get("/models/glb-model/download/source").status_code == 404


def test_downloads_require_login(pg_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    """The meshes are the dataset (NFR-7)."""
    monkeypatch.setattr(db, "get_engine", lambda: pg_engine)
    anonymous = TestClient(api.app)
    assert anonymous.get("/models/glb-model/download/source").status_code == 401
    assert anonymous.get("/models/glb-model/download/normalized").status_code == 401
