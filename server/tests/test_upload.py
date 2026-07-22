"""Tests for admin data upload (FR-9).

The upload replaces the download stage: the file *is* the raw mesh, so what
matters is that it lands under `raw/` with the right extension, gets a model row
the rest of the pipeline can act on, and reaches the convert topic — and that
anything unusable is refused *here*, not three stages later in a dead-letter.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import trimesh
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select, text

from app import api, config, db
from app.models import DownloadStatus, Model, User, UserRole
from app.security import CSRF_COOKIE, CSRF_HEADER, hash_password

ADMIN_EMAIL = "admin@imagegenie.dev"
VIEWER_EMAIL = "viewer@imagegenie.dev"
PASSWORD = "genie-secret"


def _mesh_bytes(file_type: str) -> bytes:
    """A real, loadable mesh — trimesh returns str for text formats, bytes for binary."""
    exported = trimesh.creation.box().export(file_type=file_type)
    return exported.encode() if isinstance(exported, str) else exported


@pytest.fixture
def published(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Capture convert-topic publishes instead of reaching for Pub/Sub."""
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(api, "publish_next", lambda topic, uid: calls.append((topic, uid)))
    return calls


@pytest.fixture
def upload_client(
    pg_engine: Engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    """Admin client with storage pointed at a temp dir, so blobs are inspectable."""
    monkeypatch.setattr(db, "get_engine", lambda: pg_engine)
    monkeypatch.setattr(
        api, "get_settings", lambda: config.Settings(storage_root=tmp_path)
    )
    with pg_engine.begin() as connection:
        connection.execute(
            text("TRUNCATE session, label, artifact, model, app_user RESTART IDENTITY CASCADE")
        )
    with db.session_scope() as session:
        for email, role in ((ADMIN_EMAIL, UserRole.admin), (VIEWER_EMAIL, UserRole.user)):
            session.add(
                User(
                    email=email, role=role, password_hash=hash_password(PASSWORD), verified=True
                )
            )

    client = TestClient(api.app)
    client.post("/auth/login", json={"email": ADMIN_EMAIL, "password": PASSWORD})
    client.headers[CSRF_HEADER] = client.cookies[CSRF_COOKIE]
    return client


def _upload(client: TestClient, filename: str, data: bytes):
    return client.post("/models/upload", files={"file": (filename, data)})


@pytest.mark.parametrize(
    ("filename", "file_type"),
    [("chair.glb", "glb"), ("chair.stl", "stl"), ("chair.obj", "obj")],
)
def test_upload_accepts_every_supported_format(
    upload_client: TestClient, tmp_path: Path, published, filename: str, file_type: str
) -> None:
    response = _upload(upload_client, filename, _mesh_bytes(file_type))

    assert response.status_code == 201
    uid = response.json()["uid"]

    # The blob keeps the format's own extension — convert reads it from the key.
    suffix = Path(filename).suffix
    assert (tmp_path / "raw" / f"{uid}{suffix}").is_file()

    with db.session_scope() as session:
        model = session.get(Model, uid)
        assert model.raw_key == f"raw/{uid}{suffix}"
        assert model.download_status == DownloadStatus.downloaded
        assert model.content_hash  # sha256 recorded, as the download worker does

    # Enters the pipeline at convert, since there is nothing to download.
    assert published == [(config.Settings().convert_topic, uid)]


def test_upload_titles_the_model_from_the_filename(
    upload_client: TestClient, published
) -> None:
    """The uid is random hex, so the filename is the only human-readable handle."""
    response = _upload(upload_client, "wooden chair.glb", _mesh_bytes("glb"))

    assert response.status_code == 201
    assert response.json()["title"] == "wooden chair"


def test_upload_rejects_unsupported_format(upload_client: TestClient, published) -> None:
    """FBX is refused up front rather than dead-lettering inside convert."""
    response = _upload(upload_client, "model.fbx", b"fbx-ish bytes")

    assert response.status_code == 415
    assert "fbx" in response.json()["detail"].lower()
    assert published == []  # nothing entered the pipeline


def test_upload_rejects_a_file_with_no_extension(upload_client: TestClient) -> None:
    assert _upload(upload_client, "mesh", _mesh_bytes("glb")).status_code == 415


def test_upload_rejects_corrupt_mesh(upload_client: TestClient, published) -> None:
    """A right-named file that isn't a mesh fails now, with a reason attached."""
    response = _upload(upload_client, "broken.glb", b"not actually a glb")

    assert response.status_code == 422
    assert published == []


def test_upload_rejects_empty_file(upload_client: TestClient) -> None:
    assert _upload(upload_client, "empty.glb", b"").status_code == 400


def test_upload_rejects_oversized_file(
    upload_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, published
) -> None:
    """The cap is enforced while reading, so an oversized body isn't fully buffered."""
    monkeypatch.setattr(
        api,
        "get_settings",
        lambda: config.Settings(storage_root=tmp_path, upload_max_bytes=1024),
    )

    response = _upload(upload_client, "big.glb", b"x" * 4096)

    assert response.status_code == 413
    assert "limit" in response.json()["detail"]
    assert published == []


def test_upload_requires_admin(upload_client: TestClient, published) -> None:
    """FR-8/NFR-7: a normal user may view but not feed the pipeline."""
    upload_client.post("/auth/logout")
    upload_client.post("/auth/login", json={"email": VIEWER_EMAIL, "password": PASSWORD})
    upload_client.headers[CSRF_HEADER] = upload_client.cookies[CSRF_COOKIE]

    response = _upload(upload_client, "chair.glb", _mesh_bytes("glb"))

    assert response.status_code == 403
    assert published == []


def test_upload_requires_a_csrf_token(upload_client: TestClient, published) -> None:
    """The middleware fails closed, so a new write route is covered on day one."""
    del upload_client.headers[CSRF_HEADER]

    response = _upload(upload_client, "chair.glb", _mesh_bytes("glb"))

    assert response.status_code == 403
    assert published == []


def test_uploaded_model_is_listed_and_unlabeled(
    upload_client: TestClient, published
) -> None:
    """An upload is an ordinary model from the UI's point of view — just unlabeled."""
    uid = _upload(upload_client, "chair.glb", _mesh_bytes("glb")).json()["uid"]

    body = upload_client.get("/models").json()

    listed = {item["uid"]: item for item in body["items"]}
    assert uid in listed
    assert listed[uid]["class_name"] is None  # no weak label; a human labels it


def test_two_uploads_of_the_same_file_are_distinct_models(
    upload_client: TestClient, published
) -> None:
    """Uids are generated, not content-derived, so re-upload doesn't collide."""
    data = _mesh_bytes("glb")

    first = _upload(upload_client, "chair.glb", data).json()["uid"]
    second = _upload(upload_client, "chair.glb", data).json()["uid"]

    assert first != second
    with db.session_scope() as session:
        assert session.execute(select(Model.uid)).scalars().all().count(first) == 1


def test_corrupt_mesh_error_does_not_leak_parser_internals(
    upload_client: TestClient, published
) -> None:
    """An admin can't act on 'buffer size must be a multiple of element size'.

    The real parser error goes to the log; the response says what to do about it.
    """
    response = _upload(upload_client, "broken.glb", b"not actually a glb")

    detail = response.json()["detail"]
    assert response.status_code == 422
    assert "could not read this file as GLB" in detail
    assert "buffer size" not in detail


def test_mesh_with_no_geometry_reports_the_real_reason(
    upload_client: TestClient, published
) -> None:
    """'No usable geometry' is actionable, so that one is passed through."""
    empty = trimesh.Trimesh()  # valid PLY container, zero faces
    response = _upload(upload_client, "empty-mesh.ply", empty.export(file_type="ply"))
    # .ply isn't an accepted upload format, so use a supported container instead.
    assert response.status_code == 415

    response = _upload(upload_client, "empty-mesh.stl", empty.export(file_type="stl"))
    assert response.status_code == 422
    assert "no usable geometry" in response.json()["detail"]
