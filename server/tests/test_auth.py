import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text

from app import api, db
from app.models import User, UserRole
from app.security import (
    CSRF_COOKIE,
    CSRF_HEADER,
    SESSION_COOKIE,
    csrf_tokens_match,
    generate_csrf_token,
    hash_password,
)


@pytest.fixture
def client(pg_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """API client on the test Postgres, seeded with a verified admin + an unverified user."""
    monkeypatch.setattr(db, "get_engine", lambda: pg_engine)
    with pg_engine.begin() as connection:
        connection.execute(
            text("TRUNCATE session, label, artifact, model, app_user RESTART IDENTITY CASCADE")
        )
    with db.session_scope() as session:
        session.add(
            User(
                email="admin@imagegenie.dev",
                role=UserRole.admin,
                password_hash=hash_password("genie-admin"),
                verified=True,
            )
        )
        session.add(
            User(
                email="unverified@imagegenie.dev",
                role=UserRole.user,
                password_hash=hash_password("secret1234"),
                verified=False,
            )
        )
    return TestClient(api.app)


def test_login_me_logout_cycle(client: TestClient) -> None:
    assert client.get("/auth/me").status_code == 401  # no session yet

    login = client.post(
        "/auth/login", json={"email": "admin@imagegenie.dev", "password": "genie-admin"}
    )
    assert login.status_code == 200
    assert login.json() == {"email": "admin@imagegenie.dev", "role": "admin"}

    # The client carries the httpOnly session cookie forward.
    me = client.get("/auth/me")
    assert me.status_code == 200
    assert me.json()["role"] == "admin"

    # Logout is a state change, so it carries the CSRF header like any write.
    logout = client.post("/auth/logout", headers={CSRF_HEADER: client.cookies[CSRF_COOKIE]})
    assert logout.status_code == 204
    assert client.get("/auth/me").status_code == 401  # session revoked


def test_login_sets_the_cookie_pair(client: TestClient) -> None:
    client.post("/auth/login", json={"email": "admin@imagegenie.dev", "password": "genie-admin"})
    assert client.cookies[SESSION_COOKIE]
    assert client.cookies[CSRF_COOKIE]


def test_csrf_cookie_is_readable_but_session_is_not(client: TestClient) -> None:
    """The CSRF cookie must NOT be httpOnly — the page JS has to echo it back."""
    response = client.post(
        "/auth/login", json={"email": "admin@imagegenie.dev", "password": "genie-admin"}
    )
    set_cookies = response.headers.get_list("set-cookie")
    session_header = next(header for header in set_cookies if header.startswith(SESSION_COOKIE))
    csrf_header = next(header for header in set_cookies if header.startswith(CSRF_COOKIE))
    assert "httponly" in session_header.lower()
    assert "httponly" not in csrf_header.lower()


def test_wrong_password(client: TestClient) -> None:
    response = client.post(
        "/auth/login", json={"email": "admin@imagegenie.dev", "password": "nope"}
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "invalid_credentials"


def test_unverified_cannot_log_in(client: TestClient) -> None:
    response = client.post(
        "/auth/login", json={"email": "unverified@imagegenie.dev", "password": "secret1234"}
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "unverified"


# ── CSRF double-submit (server.md#csrf) ─────────────────────────────────────
def test_csrf_tokens_are_distinct_and_high_entropy() -> None:
    tokens_set = {generate_csrf_token() for _ in range(100)}
    assert len(tokens_set) == 100
    assert all(len(token) >= 32 for token in tokens_set)


@pytest.mark.parametrize(
    ("cookie_value", "header_value"),
    [
        ("token", "token-longer"),  # length mismatch must not raise
        ("token", "other"),  # same length, different value
        ("token", None),  # header missing — the cross-site case
        (None, "token"),  # cookie missing
        (None, None),
        ("", ""),  # empty pair must not count as a match
    ],
)
def test_csrf_mismatches_are_rejected(cookie_value, header_value) -> None:
    assert not csrf_tokens_match(cookie_value, header_value)


def test_csrf_match_accepts_the_echoed_token() -> None:
    token = generate_csrf_token()
    assert csrf_tokens_match(token, token)


def test_write_without_csrf_header_is_refused(client: TestClient) -> None:
    """The core attack shape: a valid session cookie, but no header to echo.

    A cross-site page can make the browser send the cookie; it cannot read the
    cookie's value, so it cannot set the matching header.
    """
    client.post("/auth/login", json={"email": "admin@imagegenie.dev", "password": "genie-admin"})
    response = client.post("/auth/logout")  # cookie rides along, header absent
    assert response.status_code == 403
    assert response.json()["detail"] == "csrf_failure"
    assert client.get("/auth/me").status_code == 200  # still logged in


def test_write_with_wrong_csrf_header_is_refused(client: TestClient) -> None:
    client.post("/auth/login", json={"email": "admin@imagegenie.dev", "password": "genie-admin"})
    response = client.post("/auth/logout", headers={CSRF_HEADER: generate_csrf_token()})
    assert response.status_code == 403


def test_login_is_csrf_exempt_and_reads_are_unaffected(client: TestClient) -> None:
    # Login can't require a token — it's what mints one.
    assert client.post(
        "/auth/login", json={"email": "admin@imagegenie.dev", "password": "genie-admin"}
    ).status_code == 200
    # Safe methods never need the header.
    assert client.get("/auth/me").status_code == 200
    assert client.get("/healthz").status_code == 200
