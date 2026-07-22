"""Dead-letter recording, listing, and replay (server.md#dead-letters)."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select, text

from app import api, db, dead_letters
from app.dead_letters import list_dead_letters, record_failure, replay
from app.models import DeadLetter, PipelineStage, User, UserRole
from app.security import CSRF_COOKIE, CSRF_HEADER, hash_password

ADMIN_EMAIL = "admin@imagegenie.dev"
VIEWER_EMAIL = "viewer@imagegenie.dev"
PASSWORD = "genie-secret"


@pytest.fixture
def published(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Capture republished jobs instead of talking to Pub/Sub."""
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        dead_letters, "publish_next", lambda topic, uid: sent.append((topic, uid))
    )
    return sent


@pytest.fixture
def anon_client(pg_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(db, "get_engine", lambda: pg_engine)
    with pg_engine.begin() as connection:
        connection.execute(
            text(
                "TRUNCATE dead_letter, email_verification, invite, session, label,"
                " artifact, model, app_user RESTART IDENTITY CASCADE"
            )
        )
    with db.session_scope() as session:
        for email, role in ((ADMIN_EMAIL, UserRole.admin), (VIEWER_EMAIL, UserRole.user)):
            session.add(
                User(
                    email=email,
                    role=role,
                    password_hash=hash_password(PASSWORD),
                    verified=True,
                )
            )
    return TestClient(api.app)


def _login(client: TestClient, email: str) -> TestClient:
    client.post("/auth/login", json={"email": email, "password": PASSWORD})
    client.headers[CSRF_HEADER] = client.cookies[CSRF_COOKIE]
    return client


@pytest.fixture
def admin_client(anon_client: TestClient) -> TestClient:
    return _login(anon_client, ADMIN_EMAIL)


# ── Recording ───────────────────────────────────────────────────────────────
def test_records_the_error_and_attempt(anon_client) -> None:
    with db.session_scope() as session:
        record_failure(
            session, "uid-a", PipelineStage.download, "ReadTimeout fetching mesh", 4
        )

    with db.session_scope() as session:
        row = session.scalars(select(DeadLetter)).one()
        assert row.model_uid == "uid-a"
        assert row.stage is PipelineStage.download
        assert row.error == "ReadTimeout fetching mesh"
        assert row.delivery_attempt == 4


def test_refailing_updates_rather_than_piling_up(anon_client) -> None:
    """At-least-once delivery means the same job fails repeatedly; the admin
    wants current state, not one row per attempt."""
    with db.session_scope() as session:
        record_failure(session, "uid-a", PipelineStage.convert, "first error", 1)
    with db.session_scope() as session:
        record_failure(session, "uid-a", PipelineStage.convert, "later error", 5)

    with db.session_scope() as session:
        row = session.scalars(select(DeadLetter)).one()  # still exactly one
        assert row.error == "later error"
        assert row.delivery_attempt == 5


def test_the_same_uid_can_fail_in_two_stages(anon_client) -> None:
    with db.session_scope() as session:
        record_failure(session, "uid-a", PipelineStage.convert, "bad mesh")
        record_failure(session, "uid-a", PipelineStage.render, "no GL context")
    with db.session_scope() as session:
        assert len(session.scalars(select(DeadLetter)).all()) == 2


def test_a_long_error_is_truncated(anon_client) -> None:
    """Mesh libraries throw kilobyte tracebacks; the list only needs enough to
    recognise the failure."""
    with db.session_scope() as session:
        record_failure(session, "uid-a", PipelineStage.render, "x" * 10_000)
    with db.session_scope() as session:
        assert len(session.scalars(select(DeadLetter)).one().error) == (
            dead_letters.MAX_ERROR_LENGTH
        )


# ── Replay ──────────────────────────────────────────────────────────────────
def test_replay_republishes_to_the_stage_topic(anon_client, published) -> None:
    with db.session_scope() as session:
        record_failure(session, "uid-a", PipelineStage.normalize, "boom")
    with db.session_scope() as session:
        row_id = session.scalars(select(DeadLetter)).one().id

    with db.session_scope() as session:
        replay(session, row_id)

    assert published == [("normalize-jobs", "uid-a")]
    with db.session_scope() as session:
        assert session.get(DeadLetter, row_id).replayed_at is not None


def test_replayed_rows_are_hidden_but_kept(anon_client, published) -> None:
    """Kept so an admin can see they already retried it — hidden so the
    outstanding list stays actionable."""
    with db.session_scope() as session:
        record_failure(session, "uid-a", PipelineStage.download, "boom")
    with db.session_scope() as session:
        replay(session, session.scalars(select(DeadLetter)).one().id)

    with db.session_scope() as session:
        assert list_dead_letters(session, include_replayed=False) == []
        assert len(list_dead_letters(session, include_replayed=True)) == 1


def test_failing_again_after_a_replay_makes_it_outstanding(anon_client, published) -> None:
    with db.session_scope() as session:
        record_failure(session, "uid-a", PipelineStage.download, "boom")
    with db.session_scope() as session:
        replay(session, session.scalars(select(DeadLetter)).one().id)
    with db.session_scope() as session:
        record_failure(session, "uid-a", PipelineStage.download, "boom again")

    with db.session_scope() as session:
        outstanding = list_dead_letters(session, include_replayed=False)
        assert [row.error for row in outstanding] == ["boom again"]


# ── API ─────────────────────────────────────────────────────────────────────
def test_admin_lists_and_retries(admin_client: TestClient, published) -> None:
    with db.session_scope() as session:
        record_failure(session, "uid-a", PipelineStage.download, "ReadTimeout", 4)

    body = admin_client.get("/dead-letters").json()
    assert len(body) == 1
    assert body[0]["uid"] == "uid-a"
    assert body[0]["stage"] == "download"
    assert body[0]["error"] == "ReadTimeout"

    assert admin_client.post(f"/dead-letters/{body[0]['id']}/retry").status_code == 204
    assert published == [("download-jobs", "uid-a")]
    assert admin_client.get("/dead-letters").json() == []  # no longer outstanding


def test_retrying_an_unknown_row_is_404(admin_client: TestClient, published) -> None:
    assert admin_client.post("/dead-letters/999/retry").status_code == 404


def test_a_normal_user_cannot_see_failures(anon_client: TestClient) -> None:
    """Operational detail, and retry re-enqueues real pipeline work."""
    viewer = _login(anon_client, VIEWER_EMAIL)
    assert viewer.get("/dead-letters").status_code == 403
    assert viewer.post("/dead-letters/1/retry").status_code == 403


def test_anonymous_access_is_refused(anon_client: TestClient) -> None:
    assert anon_client.get("/dead-letters").status_code == 401
