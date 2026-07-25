"""Training-run dashboard read API (FR-6, web.md#training-dashboard).

Read-only and login-gated: the training script writes these rows directly
(ml/train.py), so the API only lists and drills into them.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text

from app import api, db
from app.models import TrainingMetric, TrainingRun, TrainingStatus, User, UserRole
from app.security import CSRF_COOKIE, CSRF_HEADER, hash_password

ADMIN_EMAIL = "admin@imagegenie.dev"
VIEWER_EMAIL = "viewer@imagegenie.dev"
PASSWORD = "genie-secret"


@pytest.fixture
def anon_client(pg_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(db, "get_engine", lambda: pg_engine)
    with pg_engine.begin() as connection:
        connection.execute(
            text(
                "TRUNCATE training_metric, training_run, session, app_user"
                " RESTART IDENTITY CASCADE"
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
def viewer_client(anon_client: TestClient) -> TestClient:
    """A normal (non-admin) user — the dashboard is viewable by any authed user."""
    return _login(anon_client, VIEWER_EMAIL)


def _seed_run(
    *,
    config: dict,
    data_snapshot: dict,
    status: TrainingStatus = TrainingStatus.completed,
    losses: list[tuple[int, float, float | None]] | None = None,
) -> int:
    """Insert a run plus its metric points; return the run id."""
    with db.session_scope() as session:
        run = TrainingRun(config=config, data_snapshot=data_snapshot, status=status)
        session.add(run)
        session.flush()
        for step, loss, val_loss in losses or []:
            session.add(
                TrainingMetric(run_id=run.id, step=step, loss=loss, val_loss=val_loss)
            )
        return run.id


def test_list_is_newest_first_with_headline(viewer_client: TestClient) -> None:
    first = _seed_run(
        config={"arch": "mvcnn"},
        data_snapshot={"label_count": 5},
        losses=[(0, 2.0, None), (10, 1.0, 1.2)],
    )
    second = _seed_run(
        config={"arch": "resnet50"},
        data_snapshot={"label_count": 9},
        losses=[(0, 3.0, None), (5, 0.5, None)],
    )

    body = viewer_client.get("/training-runs").json()
    assert [row["id"] for row in body] == [second, first]  # newest (highest id) first
    top = body[0]
    assert top["arch"] == "resnet50"
    assert top["label_count"] == 9
    assert top["final_loss"] == pytest.approx(0.5)  # loss at the last logged step


def test_run_with_no_metrics_has_null_final_loss(viewer_client: TestClient) -> None:
    run_id = _seed_run(
        config={"arch": "mvcnn"},
        data_snapshot={"label_count": 0},
        status=TrainingStatus.running,
    )
    row = next(r for r in viewer_client.get("/training-runs").json() if r["id"] == run_id)
    assert row["final_loss"] is None
    assert row["status"] == "running"
    assert row["finished_at"] is None


def test_detail_returns_the_blobs(viewer_client: TestClient) -> None:
    run_id = _seed_run(
        config={"arch": "mvcnn", "epochs": 20},
        data_snapshot={"label_count": 5, "label_hash": "sha256:abc"},
    )
    body = viewer_client.get(f"/training-runs/{run_id}").json()
    assert body["config"]["epochs"] == 20
    assert body["data_snapshot"]["label_hash"] == "sha256:abc"
    assert body["metrics"] is None  # not evaluated yet (B4/M7)
    assert body["status"] == "completed"


def test_detail_unknown_run_is_404(viewer_client: TestClient) -> None:
    assert viewer_client.get("/training-runs/999").status_code == 404


def test_metrics_are_in_step_order(viewer_client: TestClient) -> None:
    run_id = _seed_run(
        config={},
        data_snapshot={},
        losses=[(10, 1.0, 1.1), (0, 2.0, None), (5, 1.5, None)],  # inserted out of order
    )
    body = viewer_client.get(f"/training-runs/{run_id}/metrics").json()
    assert [point["step"] for point in body] == [0, 5, 10]
    assert body[0]["val_loss"] is None
    assert body[2]["val_loss"] == pytest.approx(1.1)


def test_metrics_unknown_run_is_404(viewer_client: TestClient) -> None:
    assert viewer_client.get("/training-runs/999/metrics").status_code == 404


def test_anonymous_access_is_refused(anon_client: TestClient) -> None:
    assert anon_client.get("/training-runs").status_code == 401
    assert anon_client.get("/training-runs/1").status_code == 401
    assert anon_client.get("/training-runs/1/metrics").status_code == 401
