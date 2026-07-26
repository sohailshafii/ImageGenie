"""Downloading meshes and trained weights (server.md#downloads).

The interesting cases are the *absent* ones. A model part-way through the
pipeline has no normalized mesh, one whose raw blob was never recorded has no
source mesh, and a still-running training run has no checkpoint — all must read
as "not available" rather than as a server error, and none may leak a
soft-deleted model.

The two gates differ deliberately and are pinned here: meshes are login-gated
(matching `/artifacts/{key}`), weights are admin-only (NFR-6).
"""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text

from app import api, db
from app.artifact_keys import normalized_key, raw_key, weights_key
from app.models import (
    DownloadStatus,
    Model,
    TrainingRun,
    TrainingStatus,
    User,
    UserRole,
)
from app.security import CSRF_COOKIE, CSRF_HEADER, hash_password

VIEWER_EMAIL = "viewer@imagegenie.dev"
ADMIN_EMAIL = "admin@imagegenie.dev"
PASSWORD = "genie-secret"


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
def anon_client(pg_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(db, "get_engine", lambda: pg_engine)
    with pg_engine.begin() as connection:
        connection.execute(
            text(
                "TRUNCATE training_metric, training_run, email_verification, invite,"
                " session, label, artifact, model, app_user RESTART IDENTITY CASCADE"
            )
        )
    with db.session_scope() as session:
        for email, role in ((VIEWER_EMAIL, UserRole.user), (ADMIN_EMAIL, UserRole.admin)):
            session.add(
                User(
                    email=email,
                    role=role,
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
    return TestClient(api.app)


def _login(client: TestClient, email: str) -> TestClient:
    client.post("/auth/login", json={"email": email, "password": PASSWORD})
    client.headers[CSRF_HEADER] = client.cookies[CSRF_COOKIE]
    return client


@pytest.fixture
def client(anon_client: TestClient) -> TestClient:
    """A **viewer** session — mesh downloads are login-gated, not admin-gated."""
    return _login(anon_client, VIEWER_EMAIL)


@pytest.fixture
def admin_client(anon_client: TestClient) -> TestClient:
    return _login(anon_client, ADMIN_EMAIL)


def _seed_run(*, status: TrainingStatus, weights_uri: str | None) -> int:
    with db.session_scope() as session:
        run = TrainingRun(
            config={"arch": "mvcnn"},
            data_snapshot={"label_count": 3},
            status=status,
            weights_uri=weights_uri,
        )
        session.add(run)
        session.flush()
        return run.id


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


def test_downloads_require_login(anon_client: TestClient, storage: FakeStorage) -> None:
    """The meshes are the dataset (NFR-7)."""
    assert anon_client.get("/models/glb-model/download/source").status_code == 401
    assert anon_client.get("/models/glb-model/download/normalized").status_code == 401


# ── Trained weights ─────────────────────────────────────────────────────────


def test_weights_download(admin_client: TestClient, storage: FakeStorage) -> None:
    run_id = _seed_run(status=TrainingStatus.completed, weights_uri=None)
    key = weights_key(run_id)
    storage.put_bytes(key, b"torch-checkpoint")
    with db.session_scope() as session:
        session.get(TrainingRun, run_id).weights_uri = key

    response = admin_client.get(f"/training-runs/{run_id}/weights")

    assert response.status_code == 200
    assert response.content == b"torch-checkpoint"
    assert (
        response.headers["content-disposition"]
        == f'attachment; filename="imagegenie-run-{run_id}.pt"'
    )


def test_a_viewer_cannot_download_weights(
    client: TestClient, storage: FakeStorage
) -> None:
    """A trained model is the artifact NFR-6 calls non-redistributable, so this
    is the one part of the training dashboard a viewer cannot reach."""
    run_id = _seed_run(status=TrainingStatus.completed, weights_uri=weights_key(1))
    storage.put_bytes(weights_key(1), b"torch-checkpoint")

    assert client.get(f"/training-runs/{run_id}/weights").status_code == 403
    # …while the rest of the run stays readable.
    assert client.get(f"/training-runs/{run_id}").status_code == 200


def test_weights_require_login(anon_client: TestClient, storage: FakeStorage) -> None:
    run_id = _seed_run(status=TrainingStatus.completed, weights_uri=weights_key(1))
    assert anon_client.get(f"/training-runs/{run_id}/weights").status_code == 401


def test_a_run_with_no_checkpoint_is_404(
    admin_client: TestClient, storage: FakeStorage
) -> None:
    """Still running, or failed before its first epoch finished."""
    run_id = _seed_run(status=TrainingStatus.running, weights_uri=None)
    assert admin_client.get(f"/training-runs/{run_id}/weights").status_code == 404


def test_a_recorded_checkpoint_whose_blob_is_gone_is_404(
    admin_client: TestClient, storage: FakeStorage
) -> None:
    run_id = _seed_run(status=TrainingStatus.completed, weights_uri=weights_key(1))
    assert admin_client.get(f"/training-runs/{run_id}/weights").status_code == 404


def test_unknown_run_is_404(admin_client: TestClient, storage: FakeStorage) -> None:
    assert admin_client.get("/training-runs/9999/weights").status_code == 404
