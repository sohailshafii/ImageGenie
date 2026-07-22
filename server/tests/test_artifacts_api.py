"""Serving pipeline artifacts to the labeling UI (server.md#serving-artifacts).

This is what makes the UI usable — without it the viewer shows a placeholder and
the grid shows emoji. The cases that matter are the partial ones: a model part-way
through the pipeline must degrade to fewer views, not to broken images.
"""

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text

from app import api, db
from app.artifact_keys import NUM_VIEWS, normalized_key, view_key
from app.models import DownloadStatus, Model, User, UserRole
from app.security import CSRF_COOKIE, CSRF_HEADER, hash_password

ADMIN_EMAIL = "admin@imagegenie.dev"
PASSWORD = "genie-admin"


class FakeStorage:
    """In-memory Storage. `signable` mirrors GCS; False mirrors local dev."""

    def __init__(self, signable: bool) -> None:
        self.blobs: dict[str, bytes] = {}
        self._signable = signable

    def exists(self, key: str) -> bool:
        return key in self.blobs

    def put_bytes(self, key: str, data: bytes) -> None:
        self.blobs[key] = data

    def get_bytes(self, key: str) -> bytes:
        return self.blobs[key]

    def signed_url(self, key: str, ttl: timedelta) -> str | None:
        if not self._signable:
            return None
        return f"https://signed.test/{key}?ttl={int(ttl.total_seconds())}"


@pytest.fixture
def storage(monkeypatch: pytest.MonkeyPatch) -> FakeStorage:
    """Signing backend (the cloud path) by default."""
    fake = FakeStorage(signable=True)
    monkeypatch.setattr(api, "build_storage", lambda _settings: fake)
    return fake


@pytest.fixture
def client(pg_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> TestClient:
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
                email=ADMIN_EMAIL,
                role=UserRole.admin,
                password_hash=hash_password(PASSWORD),
                verified=True,
            )
        )
        session.add(Model(uid="rendered", download_status=DownloadStatus.downloaded))
        session.add(Model(uid="partial", download_status=DownloadStatus.downloaded))
        session.add(Model(uid="fresh", download_status=DownloadStatus.downloaded))
    test_client = TestClient(api.app)
    test_client.post("/auth/login", json={"email": ADMIN_EMAIL, "password": PASSWORD})
    test_client.headers[CSRF_HEADER] = test_client.cookies[CSRF_COOKIE]
    return test_client


def _fill_renders(storage: FakeStorage, uid: str, count: int) -> None:
    for index in range(count):
        storage.put_bytes(view_key(uid, index), b"png-bytes")


def test_returns_every_view_and_the_mesh(client: TestClient, storage: FakeStorage) -> None:
    _fill_renders(storage, "rendered", NUM_VIEWS)
    storage.put_bytes(normalized_key("rendered"), b"ply-bytes")

    body = client.get("/models/rendered/artifacts").json()

    assert len(body["views"]) == NUM_VIEWS
    assert body["views"][0].startswith("https://signed.test/")
    assert body["mesh"].startswith("https://signed.test/")
    assert "view_00.png" in body["views"][0]  # in view order
    assert "view_11.png" in body["views"][-1]


def test_a_part_way_model_returns_only_what_exists(
    client: TestClient, storage: FakeStorage
) -> None:
    """Rendering writes views one at a time, so a model can legitimately have
    some. Better fewer images than broken ones."""
    _fill_renders(storage, "partial", 3)

    body = client.get("/models/partial/artifacts").json()

    assert len(body["views"]) == 3
    assert body["mesh"] is None  # normalize hasn't run


def test_an_unprocessed_model_returns_empty_not_an_error(
    client: TestClient, storage: FakeStorage
) -> None:
    body = client.get("/models/fresh/artifacts").json()
    assert body == {"uid": "fresh", "views": [], "mesh": None}


def test_unknown_model_is_404(client: TestClient, storage: FakeStorage) -> None:
    assert client.get("/models/nope/artifacts").status_code == 404


def test_artifacts_require_login(pg_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    """The renders are the dataset (NFR-7)."""
    monkeypatch.setattr(db, "get_engine", lambda: pg_engine)
    anonymous = TestClient(api.app)
    assert anonymous.get("/models/rendered/artifacts").status_code == 401
    assert anonymous.get("/artifacts/processed/renders/x/view_00.png").status_code == 401


def test_falls_back_to_streaming_when_signing_is_unavailable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local dev has no signing, so URLs must point back at the API."""
    fake = FakeStorage(signable=False)
    monkeypatch.setattr(api, "build_storage", lambda _settings: fake)
    _fill_renders(fake, "rendered", 2)

    body = client.get("/models/rendered/artifacts").json()

    assert body["views"][0] == "/artifacts/processed/renders/rendered/view_00.png"
    # …and that URL actually serves the bytes.
    streamed = client.get(body["views"][0])
    assert streamed.status_code == 200
    assert streamed.content == b"png-bytes"
    assert streamed.headers["content-type"] == "image/png"


def test_streaming_a_missing_key_is_404(client: TestClient, storage: FakeStorage) -> None:
    assert client.get("/artifacts/processed/renders/nope/view_00.png").status_code == 404


def test_streaming_rejects_traversal(client: TestClient, storage: FakeStorage) -> None:
    response = client.get("/artifacts/processed/../../etc/passwd")
    assert response.status_code in (400, 404)  # normalized away or refused outright
