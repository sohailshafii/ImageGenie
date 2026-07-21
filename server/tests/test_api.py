import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text

from app import api, db
from app.models import DownloadStatus, Label, LabelSource, Model


@pytest.fixture
def client(pg_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """API client backed by the test Postgres, seeded with two labeled models."""
    monkeypatch.setattr(db, "get_engine", lambda: pg_engine)
    with pg_engine.begin() as connection:
        connection.execute(
            text("TRUNCATE label, artifact, model, app_user RESTART IDENTITY CASCADE")
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
    return TestClient(api.app)


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
    }
    # And it sticks on the next read.
    assert client.get("/models/m2").json()["source"] == "manual"
