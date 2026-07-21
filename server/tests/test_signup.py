"""Invite-gated signup, email verification, and admin invites (web.md#auth--roles).

Mirrors the flows the frontend mock (`web/src/api/auth.ts`) already implements, so
swapping the mock for this API doesn't change any component.
"""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select, text

from app import api, db
from app.models import EmailVerification, Invite, User, UserRole
from app.security import CSRF_COOKIE, CSRF_HEADER, hash_password, hash_token

ADMIN_EMAIL = "admin@imagegenie.dev"
ADMIN_PASSWORD = "genie-admin"
INVITED_EMAIL = "labeler@imagegenie.dev"
NEW_PASSWORD = "long-enough-password"


@pytest.fixture
def anon_client(pg_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Logged-out client, seeded with one admin and one open invite."""
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
                password_hash=hash_password(ADMIN_PASSWORD),
                verified=True,
            )
        )
    return TestClient(api.app)


@pytest.fixture
def admin_client(anon_client: TestClient) -> TestClient:
    response = anon_client.post(
        "/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}
    )
    assert response.status_code == 200
    anon_client.headers[CSRF_HEADER] = anon_client.cookies[CSRF_COOKIE]
    return anon_client


def _open_invite(email: str = INVITED_EMAIL) -> None:
    """Seed an open invite directly, for tests not exercising the admin route."""
    with db.session_scope() as session:
        session.add(
            Invite(email=email, expires_at=datetime.now(UTC) + timedelta(days=14))
        )


# ── Signup ──────────────────────────────────────────────────────────────────
def test_signup_requires_an_open_invite(anon_client: TestClient) -> None:
    response = anon_client.post(
        "/auth/signup", json={"email": "stranger@nowhere.dev", "password": NEW_PASSWORD}
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "invite_required"


def test_signup_creates_an_unverified_user_and_spends_the_invite(
    anon_client: TestClient,
) -> None:
    _open_invite()
    assert anon_client.post(
        "/auth/signup", json={"email": INVITED_EMAIL, "password": NEW_PASSWORD}
    ).status_code == 204

    with db.session_scope() as session:
        user = session.scalar(select(User).where(User.email == INVITED_EMAIL))
        assert user is not None
        assert user.verified is False  # must verify before logging in
        assert user.role is UserRole.user  # an invite never grants admin
        assert session.get(Invite, INVITED_EMAIL).accepted is True


def test_unverified_account_cannot_log_in_yet(anon_client: TestClient) -> None:
    _open_invite()
    anon_client.post("/auth/signup", json={"email": INVITED_EMAIL, "password": NEW_PASSWORD})
    response = anon_client.post(
        "/auth/login", json={"email": INVITED_EMAIL, "password": NEW_PASSWORD}
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "unverified"


def test_signup_rejects_a_short_password(anon_client: TestClient) -> None:
    _open_invite()
    response = anon_client.post("/auth/signup", json={"email": INVITED_EMAIL, "password": "short"})
    assert response.status_code == 400
    assert response.json()["detail"] == "validation_error"


def test_a_spent_invite_cannot_be_reused(anon_client: TestClient) -> None:
    _open_invite()
    anon_client.post("/auth/signup", json={"email": INVITED_EMAIL, "password": NEW_PASSWORD})
    # Same address again — the account now exists.
    assert anon_client.post(
        "/auth/signup", json={"email": INVITED_EMAIL, "password": NEW_PASSWORD}
    ).status_code == 409


def test_signup_does_not_leak_account_existence_without_an_invite(
    anon_client: TestClient,
) -> None:
    """The admin account exists, but a caller with no invite for it learns only
    `invite_required` — never `email_taken`."""
    response = anon_client.post(
        "/auth/signup", json={"email": ADMIN_EMAIL, "password": NEW_PASSWORD}
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "invite_required"


def test_expired_invite_is_refused(anon_client: TestClient) -> None:
    from datetime import UTC, datetime, timedelta

    with db.session_scope() as session:
        session.add(
            Invite(email=INVITED_EMAIL, expires_at=datetime.now(UTC) - timedelta(seconds=1))
        )
    response = anon_client.post(
        "/auth/signup", json={"email": INVITED_EMAIL, "password": NEW_PASSWORD}
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "invite_required"


# ── Email verification ──────────────────────────────────────────────────────
def _signup_and_capture_token(client: TestClient, caplog) -> str:
    """Sign up and pull the token out of the logged verification link.

    There is no mail transport yet, so the endpoint logs the link; this reads it
    the way a developer would. Replace when real email lands.
    """
    _open_invite()
    with caplog.at_level("INFO", logger="app.api"):
        client.post("/auth/signup", json={"email": INVITED_EMAIL, "password": NEW_PASSWORD})
    line = next(record for record in caplog.records if "verification link" in record.getMessage())
    return line.getMessage().split("token=")[1]


def test_verification_marks_the_account_and_enables_login(
    anon_client: TestClient, caplog
) -> None:
    token = _signup_and_capture_token(anon_client, caplog)

    assert anon_client.post("/auth/verify-email", json={"token": token}).status_code == 204
    assert anon_client.post(
        "/auth/login", json={"email": INVITED_EMAIL, "password": NEW_PASSWORD}
    ).status_code == 200


def test_verification_token_is_single_use(anon_client: TestClient, caplog) -> None:
    token = _signup_and_capture_token(anon_client, caplog)
    assert anon_client.post("/auth/verify-email", json={"token": token}).status_code == 204
    replay = anon_client.post("/auth/verify-email", json={"token": token})
    assert replay.status_code == 400
    assert replay.json()["detail"] == "invalid_token"


def test_unknown_verification_token_is_refused(anon_client: TestClient) -> None:
    response = anon_client.post("/auth/verify-email", json={"token": "not-a-real-token"})
    assert response.status_code == 400
    assert response.json()["detail"] == "invalid_token"


def test_expired_verification_token_is_refused_and_consumed(
    anon_client: TestClient, caplog
) -> None:
    from datetime import UTC, datetime, timedelta

    token = _signup_and_capture_token(anon_client, caplog)
    with db.session_scope() as session:
        record = session.get(EmailVerification, hash_token(token))
        record.expires_at = datetime.now(UTC) - timedelta(seconds=1)

    response = anon_client.post("/auth/verify-email", json={"token": token})
    assert response.status_code == 400
    assert response.json()["detail"] == "expired_token"
    with db.session_scope() as session:  # spent, not left lingering
        assert session.get(EmailVerification, hash_token(token)) is None


def test_tokens_are_stored_hashed_never_in_the_clear(
    anon_client: TestClient, caplog
) -> None:
    token = _signup_and_capture_token(anon_client, caplog)
    with db.session_scope() as session:
        stored = session.scalars(select(EmailVerification)).all()
        assert len(stored) == 1
        assert stored[0].token_hash != token
        assert stored[0].token_hash == hash_token(token)


def test_resend_replaces_the_previous_token(anon_client: TestClient, caplog) -> None:
    first = _signup_and_capture_token(anon_client, caplog)
    caplog.clear()
    with caplog.at_level("INFO", logger="app.api"):
        assert anon_client.post(
            "/auth/verify-email/resend", json={"email": INVITED_EMAIL}
        ).status_code == 204
    second = next(
        record for record in caplog.records if "verification link" in record.getMessage()
    ).getMessage().split("token=")[1]

    assert second != first
    assert anon_client.post("/auth/verify-email", json={"token": first}).status_code == 400
    assert anon_client.post("/auth/verify-email", json={"token": second}).status_code == 204


def test_resend_is_silent_about_unknown_addresses(anon_client: TestClient) -> None:
    """Always 204 — a varying status would be an account-existence oracle."""
    assert anon_client.post(
        "/auth/verify-email/resend", json={"email": "nobody@nowhere.dev"}
    ).status_code == 204
    assert anon_client.post(
        "/auth/verify-email/resend", json={"email": ADMIN_EMAIL}
    ).status_code == 204  # exists, but already verified


# ── Admin invites ───────────────────────────────────────────────────────────
def test_admin_can_mint_an_invite_that_unlocks_signup(admin_client: TestClient) -> None:
    response = admin_client.post("/auth/invites", json={"email": "fresh@imagegenie.dev"})
    assert response.status_code == 201
    assert response.json()["email"] == "fresh@imagegenie.dev"
    assert response.json()["accepted"] is False

    assert admin_client.post(
        "/auth/signup", json={"email": "fresh@imagegenie.dev", "password": NEW_PASSWORD}
    ).status_code == 204


def test_invites_are_idempotent_per_email(admin_client: TestClient) -> None:
    admin_client.post("/auth/invites", json={"email": INVITED_EMAIL})
    admin_client.post("/auth/invites", json={"email": INVITED_EMAIL})
    with db.session_scope() as session:
        assert len(session.scalars(select(Invite)).all()) == 1  # refreshed, not duplicated


def test_invite_normalizes_the_email(admin_client: TestClient) -> None:
    admin_client.post("/auth/invites", json={"email": "  MixedCase@ImageGenie.dev "})
    with db.session_scope() as session:
        assert session.get(Invite, "mixedcase@imagegenie.dev") is not None


def test_invite_records_who_sent_it(admin_client: TestClient) -> None:
    admin_client.post("/auth/invites", json={"email": INVITED_EMAIL})
    with db.session_scope() as session:
        assert session.get(Invite, INVITED_EMAIL).invited_by == ADMIN_EMAIL


def test_invite_rejects_a_malformed_email(admin_client: TestClient) -> None:
    response = admin_client.post("/auth/invites", json={"email": "not-an-email"})
    assert response.status_code == 400
    assert response.json()["detail"] == "validation_error"


def test_anonymous_cannot_invite(anon_client: TestClient) -> None:
    """403 from the CSRF layer, which answers before auth — the point is that an
    unauthenticated caller cannot mint invites."""
    assert anon_client.post("/auth/invites", json={"email": "x@y.dev"}).status_code == 403


def test_normal_user_cannot_invite(anon_client: TestClient, caplog) -> None:
    token = _signup_and_capture_token(anon_client, caplog)
    anon_client.post("/auth/verify-email", json={"token": token})
    anon_client.post("/auth/login", json={"email": INVITED_EMAIL, "password": NEW_PASSWORD})
    anon_client.headers[CSRF_HEADER] = anon_client.cookies[CSRF_COOKIE]

    response = anon_client.post("/auth/invites", json={"email": "x@y.dev"})
    assert response.status_code == 403
    assert response.json()["detail"] == "forbidden"


# ── Rate limiting on the new surfaces ───────────────────────────────────────
def test_signup_is_rate_limited_per_ip(anon_client: TestClient) -> None:
    statuses = [
        anon_client.post(
            "/auth/signup", json={"email": f"user{index}@nowhere.dev", "password": NEW_PASSWORD}
        ).status_code
        for index in range(api.SIGNUP_PER_IP.max_hits + 1)
    ]
    assert statuses[-1] == 429


def test_resend_is_rate_limited_per_email(anon_client: TestClient) -> None:
    statuses = [
        anon_client.post("/auth/verify-email/resend", json={"email": INVITED_EMAIL}).status_code
        for _ in range(api.RESEND_PER_EMAIL.max_hits + 1)
    ]
    assert statuses[-1] == 429
    assert int(
        anon_client.post(
            "/auth/verify-email/resend", json={"email": INVITED_EMAIL}
        ).headers["Retry-After"]
    ) >= 1
