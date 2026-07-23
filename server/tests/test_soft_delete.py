"""Soft delete (FR-9, admin-only): hiding a model without dropping its data.

This module covers the *read-path contract* — a soft-deleted model is invisible
to every route a labeler uses, exactly as if it were gone. The delete/restore
endpoints and the Deleted view are exercised in the endpoint tests; here the
`deleted_at` column is set directly, so the hiding is tested independently of the
routes that will set it.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import trimesh
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text

from app import api, config, db
from app.models import DownloadStatus, Model, User, UserRole
from app.security import CSRF_COOKIE, CSRF_HEADER, hash_password

ADMIN_EMAIL = "admin@imagegenie.dev"
PASSWORD = "genie-secret"


@pytest.fixture
def client(pg_engine: Engine, tmp_path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Admin client over a temp storage root, with one live and one deleted model."""
    monkeypatch.setattr(db, "get_engine", lambda: pg_engine)
    monkeypatch.setattr(api, "get_settings", lambda: config.Settings(storage_root=tmp_path))
    with pg_engine.begin() as connection:
        connection.execute(
            text("TRUNCATE session, label, artifact, model, app_user RESTART IDENTITY CASCADE")
        )
    with db.session_scope() as session:
        session.add(Model(uid="live", download_status=DownloadStatus.downloaded))
        session.add(
            Model(
                uid="gone",
                download_status=DownloadStatus.downloaded,
                deleted_at=datetime.now(UTC),
            )
        )
        session.add(
            User(
                email=ADMIN_EMAIL,
                role=UserRole.admin,
                password_hash=hash_password(PASSWORD),
                verified=True,
            )
        )
    # A rendered view for the deleted model, so the artifacts route has something
    # to find if it failed to filter.
    (tmp_path / "processed" / "renders" / "gone").mkdir(parents=True)
    (tmp_path / "processed" / "renders" / "gone" / "view_00.png").write_bytes(b"png")

    http = TestClient(api.app)
    http.post("/auth/login", json={"email": ADMIN_EMAIL, "password": PASSWORD})
    http.headers[CSRF_HEADER] = http.cookies[CSRF_COOKIE]
    return http


def test_deleted_model_is_absent_from_the_listing(client: TestClient) -> None:
    body = client.get("/models").json()
    uids = {item["uid"] for item in body["items"]}
    assert uids == {"live"}
    assert body["total"] == 1  # the count reflects the filter, not just the page


def test_deleted_model_detail_is_404(client: TestClient) -> None:
    assert client.get("/models/gone").status_code == 404
    assert client.get("/models/live").status_code == 200


def test_deleted_model_artifacts_are_404(client: TestClient) -> None:
    """Even though a render blob exists on disk, the route refuses a deleted model."""
    assert client.get("/models/gone/artifacts").status_code == 404


def test_labeling_a_deleted_model_is_404(client: TestClient) -> None:
    response = client.put("/models/gone/label", json={"class_name": "chair"})
    assert response.status_code == 404


def _mesh(file_type: str) -> bytes:
    exported = trimesh.creation.box().export(file_type=file_type)
    return exported.encode() if isinstance(exported, str) else exported


def test_upload_still_produces_a_visible_model(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guard against the filter accidentally hiding freshly-created models."""
    monkeypatch.setattr(api, "publish_next", lambda topic, uid: None)
    uid = client.post("/models/upload", files={"file": ("m.glb", _mesh("glb"))}).json()["uid"]

    assert client.get(f"/models/{uid}").status_code == 200


# ── Delete / restore / deleted-list endpoints ───────────────────────────────


def test_delete_hides_the_model_then_restore_brings_it_back(client: TestClient) -> None:
    assert client.get("/models/live").status_code == 200

    assert client.delete("/models/live").status_code == 204
    assert client.get("/models/live").status_code == 404
    assert "live" not in {item["uid"] for item in client.get("/models").json()["items"]}

    restored = client.post("/models/live/restore")
    assert restored.status_code == 200
    assert restored.json()["uid"] == "live"
    assert client.get("/models/live").status_code == 200


def test_delete_is_idempotent(client: TestClient) -> None:
    """A double click or retried request must not error on the second delete."""
    assert client.delete("/models/gone").status_code == 204  # already deleted
    assert client.delete("/models/live").status_code == 204
    assert client.delete("/models/live").status_code == 204


def test_restore_of_a_live_model_is_a_noop(client: TestClient) -> None:
    assert client.post("/models/live/restore").status_code == 200


def test_delete_and_restore_404_on_unknown_uid(client: TestClient) -> None:
    assert client.delete("/models/nope").status_code == 404
    assert client.post("/models/nope/restore").status_code == 404


def test_deleted_list_shows_only_deleted_models(client: TestClient) -> None:
    body = client.get("/models/deleted").json()
    assert {item["uid"] for item in body["items"]} == {"gone"}
    assert body["total"] == 1

    # After deleting 'live', both appear in the deleted view and neither in browse.
    client.delete("/models/live")
    deleted = {item["uid"] for item in client.get("/models/deleted").json()["items"]}
    assert deleted == {"gone", "live"}
    assert client.get("/models").json()["items"] == []


def test_deleted_route_is_not_shadowed_by_the_uid_route(client: TestClient) -> None:
    """`/models/deleted` must resolve to the list, not GET /models/{uid='deleted'}."""
    response = client.get("/models/deleted")
    assert response.status_code == 200
    assert "items" in response.json()  # a page, not a single-model 404


def _viewer(client: TestClient) -> TestClient:
    """Re-auth the shared client as a normal user (added on the fly)."""
    with db.session_scope() as session:
        session.add(
            User(
                email="viewer@imagegenie.dev",
                role=UserRole.user,
                password_hash=hash_password(PASSWORD),
                verified=True,
            )
        )
    client.post("/auth/logout")
    client.post("/auth/login", json={"email": "viewer@imagegenie.dev", "password": PASSWORD})
    client.headers[CSRF_HEADER] = client.cookies[CSRF_COOKIE]
    return client


def test_delete_requires_admin(client: TestClient) -> None:
    viewer = _viewer(client)
    assert viewer.delete("/models/live").status_code == 403
    assert viewer.post("/models/live/restore").status_code == 403
    assert viewer.get("/models/deleted").status_code == 403


def test_delete_requires_csrf_token(client: TestClient) -> None:
    del client.headers[CSRF_HEADER]
    assert client.delete("/models/live").status_code == 403
