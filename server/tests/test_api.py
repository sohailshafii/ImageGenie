import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select, text

from app import api, db
from app.models import DownloadStatus, Label, LabelSource, Model, User, UserRole
from app.security import CSRF_COOKIE, CSRF_HEADER, hash_password

ADMIN_EMAIL = "admin@imagegenie.dev"
VIEWER_EMAIL = "viewer@imagegenie.dev"
PASSWORD = "genie-secret"


@pytest.fixture
def anon_client(pg_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Logged-out API client on the test Postgres, seeded with two labeled models.

    Also seeds an admin and a normal (view-only) user for the role tests.
    """
    monkeypatch.setattr(db, "get_engine", lambda: pg_engine)
    with pg_engine.begin() as connection:
        connection.execute(
            text("TRUNCATE session, label, artifact, model, app_user RESTART IDENTITY CASCADE")
        )
    with db.session_scope() as session:
        session.add(Model(uid="m1", download_status=DownloadStatus.downloaded))
        session.add(Model(uid="m2", download_status=DownloadStatus.downloaded))
        session.flush()  # models must exist before the labels' FK references them
        # m1: a weak label, then a manual correction — the manual one is "current".
        session.add(
            Label(model_uid="m1", class_name="chair", source=LabelSource.weak, confidence=0.7)
        )
        session.add(Label(model_uid="m1", class_name="table", source=LabelSource.manual))
        # m2: weak only.
        session.add(
            Label(model_uid="m2", class_name="car", source=LabelSource.weak, confidence=0.9)
        )
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
    """Log `client` in — it carries the session cookie forward on later calls.

    Also echoes the CSRF cookie into the header on every later request, which is
    what the browser client does for unsafe methods (server.md#csrf).
    """
    response = client.post("/auth/login", json={"email": email, "password": PASSWORD})
    assert response.status_code == 200
    client.headers[CSRF_HEADER] = client.cookies[CSRF_COOKIE]
    return client


@pytest.fixture
def client(anon_client: TestClient) -> TestClient:
    """The default client: logged in as an admin, so reads *and* writes are allowed."""
    return _login(anon_client, ADMIN_EMAIL)


@pytest.fixture
def viewer_client(anon_client: TestClient) -> TestClient:
    """Logged in as a normal user — may read, may not correct labels (FR-8)."""
    return _login(anon_client, VIEWER_EMAIL)


def test_list_resolves_current_label(client: TestClient) -> None:
    body = client.get("/models").json()
    assert body["total"] == 2
    by_uid = {item["uid"]: item for item in body["items"]}
    # manual correction wins over the weak label for m1
    assert by_uid["m1"]["class_name"] == "table"
    assert by_uid["m1"]["source"] == "manual"
    assert by_uid["m2"]["class_name"] == "car"
    assert by_uid["m2"]["source"] == "weak"


def test_filters(client: TestClient) -> None:
    weak = client.get("/models", params={"source": "weak"}).json()
    assert [item["uid"] for item in weak["items"]] == ["m2"]  # only m2 is still weak
    cars = client.get("/models", params={"class_name": "car"}).json()
    assert [item["uid"] for item in cars["items"]] == ["m2"]


def test_serves_store_metadata_when_present(client: TestClient) -> None:
    with db.session_scope() as session:
        model = session.get(Model, "m1")
        model.title = "Vintage Wooden Chair"
        model.tags = ["furniture", "wood"]

    body = client.get("/models/m1").json()
    assert body["title"] == "Vintage Wooden Chair"
    assert body["tags"] == ["furniture", "wood"]


def test_falls_back_to_the_uid_before_the_metadata_backfill(client: TestClient) -> None:
    """A dull caption beats none — models are ingested long before metadata."""
    body = client.get("/models/m2").json()
    assert body["title"] == "model m2"
    assert body["tags"] == []


def test_get_one_and_404(client: TestClient) -> None:
    assert client.get("/models/m2").json()["class_name"] == "car"
    assert client.get("/models/nope").status_code == 404


def test_put_label_records_manual(client: TestClient) -> None:
    # Correct m2 to weapon → a new manual row becomes current.
    response = client.put("/models/m2/label", json={"class_name": "weapon"})
    assert response.status_code == 200
    assert response.json() == {
        "uid": "m2",
        "title": "model m2",
        "tags": [],
        "class_name": "weapon",
        "source": "manual",
        "confidence": None,
        # Emitted without checking the blob exists — see ModelSummaryOut.thumbnail.
        "thumbnail": "/artifacts/processed/renders/m2/view_00.png",
    }
    # And it sticks on the next read.
    assert client.get("/models/m2").json()["source"] == "manual"


def test_correction_is_attributed_to_the_calling_admin(client: TestClient) -> None:
    assert client.put("/models/m2/label", json={"class_name": "weapon"}).status_code == 200
    with db.session_scope() as session:
        label = session.scalars(
            select(Label).where(Label.model_uid == "m2", Label.source == LabelSource.manual)
        ).one()
        assert label.annotator == ADMIN_EMAIL


@pytest.mark.parametrize("path", ["/models", "/models/m2"])
def test_reads_require_login(anon_client: TestClient, path: str) -> None:
    assert anon_client.get(path).status_code == 401


def test_anonymous_write_is_refused(anon_client: TestClient) -> None:
    """403, not 401: the CSRF middleware runs ahead of the auth dependency.

    Not a UX regression for an expired session — the CSRF cookie shares the
    session's max-age, and a server-side revocation leaves the cookie in place,
    so that path still matches CSRF and falls through to a 401.
    """
    response = anon_client.put("/models/m2/label", json={"class_name": "weapon"})
    assert response.status_code == 403
    assert response.json()["detail"] == "csrf_failure"


def test_viewer_can_read(viewer_client: TestClient) -> None:
    assert viewer_client.get("/models").json()["total"] == 2
    assert viewer_client.get("/models/m2").json()["class_name"] == "car"


def test_viewer_cannot_correct_labels(viewer_client: TestClient) -> None:
    response = viewer_client.put("/models/m2/label", json={"class_name": "weapon"})
    assert response.status_code == 403
    # …and the weak label is untouched.
    assert viewer_client.get("/models/m2").json()["source"] == "weak"
