"""Serving the SPA and mounting the API under /api (server.md#serving-the-spa).

`root_app` is the production entrypoint: the API mounted at `/api`, the built SPA
at the root. What matters here — and can't be seen testing `app` directly — is
that the two namespaces no longer collide (`/models/{uid}` is a page at the root
and JSON under `/api`), that client-routed deep links fall back to the shell, and
that the artifact fallback URL picks up the `/api` prefix from the mount.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text

from app import api, config, db
from app.models import DownloadStatus, Model, User, UserRole
from app.security import CSRF_COOKIE, CSRF_HEADER, hash_password

ADMIN_EMAIL = "admin@imagegenie.dev"
PASSWORD = "genie-secret"


@pytest.fixture
def spa_dir(tmp_path: Path) -> Path:
    """A minimal built SPA: an index shell and one hashed asset."""
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<!doctype html><title>ImageGenie</title>", "utf-8")
    (tmp_path / "assets" / "index-abc123.js").write_text("console.log('spa')", "utf-8")
    (tmp_path / "favicon.svg").write_text("<svg/>", "utf-8")
    return tmp_path


@pytest.fixture
def client(pg_engine: Engine, spa_dir: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A client against `root_app` (API under /api, SPA at root), one admin seeded."""
    monkeypatch.setattr(db, "get_engine", lambda: pg_engine)
    monkeypatch.setattr(api, "get_settings", lambda: config.Settings(spa_dir=spa_dir))
    with pg_engine.begin() as connection:
        connection.execute(
            text("TRUNCATE session, label, artifact, model, app_user RESTART IDENTITY CASCADE")
        )
    with db.session_scope() as session:
        session.add(Model(uid="m1", download_status=DownloadStatus.downloaded))
        session.add(
            User(
                email=ADMIN_EMAIL,
                role=UserRole.admin,
                password_hash=hash_password(PASSWORD),
                verified=True,
            )
        )
    return TestClient(api.root_app)


# ── SPA serving ─────────────────────────────────────────────────────────────


def test_root_serves_the_index_shell(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "ImageGenie" in response.text
    assert response.headers["cache-control"] == "no-cache"


def test_client_routed_paths_fall_back_to_the_shell(client: TestClient) -> None:
    """The colliding names resolve to the SPA at the root — no API JSON leaks in.

    This is the whole point of the /api mount: `/models/m1` and `/dead-letters`
    are pages here, and only JSON under `/api`.
    """
    for path in ("/deleted", "/models/m1", "/dead-letters", "/upload"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert "ImageGenie" in response.text


def test_asset_is_served_with_immutable_cache(client: TestClient) -> None:
    response = client.get("/assets/index-abc123.js")
    assert response.status_code == 200
    assert "console.log" in response.text
    assert response.headers["cache-control"] == "public, max-age=31536000, immutable"


def test_top_level_file_is_served_but_not_immutably_cached(client: TestClient) -> None:
    response = client.get("/favicon.svg")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"


def test_traversal_falls_back_to_the_shell(client: TestClient) -> None:
    response = client.get("/../../etc/hosts")
    assert response.status_code == 200
    assert "ImageGenie" in response.text


# ── API under /api ──────────────────────────────────────────────────────────


def test_api_is_reachable_under_the_api_prefix(client: TestClient) -> None:
    """The same path that serves the shell at the root returns JSON under /api."""
    client.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": PASSWORD})
    client.headers[CSRF_HEADER] = client.cookies[CSRF_COOKIE]

    response = client.get("/api/models/m1")
    assert response.status_code == 200
    body = response.json()
    assert body["uid"] == "m1"


def test_artifact_fallback_url_carries_the_api_prefix(client: TestClient) -> None:
    """Mounted, the streaming-fallback URL must be `/api/artifacts/...`.

    Local storage can't sign, so the thumbnail falls back to streaming; served
    under the mount, that URL has to include `/api` or the browser would request
    it at the root and get the SPA shell instead of the image.
    """
    client.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": PASSWORD})
    client.headers[CSRF_HEADER] = client.cookies[CSRF_COOKIE]

    body = client.get("/api/models").json()
    thumbnail = body["items"][0]["thumbnail"]
    assert thumbnail.startswith("/api/artifacts/")


def test_api_404_is_json_not_the_shell(client: TestClient) -> None:
    """An unknown path *under* /api is an API 404, not the SPA fallback."""
    response = client.get("/api/models/does-not-exist")
    # 401 (auth runs first) or 404 — either way JSON from the API, not HTML.
    assert response.headers["content-type"].startswith("application/json")


def test_no_spa_dir_leaves_unknown_paths_as_404(
    pg_engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A misconfigured deploy (no spa_dir) fails loudly rather than serving nothing."""
    monkeypatch.setattr(db, "get_engine", lambda: pg_engine)
    monkeypatch.setattr(api, "get_settings", lambda: config.Settings(spa_dir=None))
    client = TestClient(api.root_app)

    assert client.get("/some/client/route").status_code == 404
