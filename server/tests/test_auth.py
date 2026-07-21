import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text

from app import api, db
from app.models import User, UserRole
from app.security import hash_password


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

    assert client.post("/auth/logout").status_code == 204
    assert client.get("/auth/me").status_code == 401  # session revoked


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
