"""Classifying a catalog model with the newest trained run.

The ml package is stubbed rather than imported: these cover the *route's*
contract — who may call it, which states are 404 vs 503, what shape comes back —
while the inference itself is covered by ml/tests/test_infer.py. Importing torch
here would also make the server suite depend on a package the API image gets
separately.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text

from app import api, db, predict
from app.models import (
    Artifact,
    ArtifactStage,
    ArtifactStatus,
    DownloadStatus,
    Model,
    TrainingRun,
    TrainingStatus,
    User,
    UserRole,
)
from app.security import CSRF_COOKIE, CSRF_HEADER, hash_password

ADMIN_EMAIL = "admin@imagegenie.dev"
VIEWER_EMAIL = "viewer@imagegenie.dev"
PASSWORD = "genie-secret"

RANKED = [("chair", 0.71), ("table", 0.2), ("lamp", 0.09)]


@pytest.fixture
def anon_client(pg_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(db, "get_engine", lambda: pg_engine)
    with pg_engine.begin() as connection:
        connection.execute(
            text(
                "TRUNCATE evaluation, training_metric, training_run, session,"
                " label, artifact, model, app_user RESTART IDENTITY CASCADE"
            )
        )
    with db.session_scope() as session:
        # m1 is rendered; m2 never was.
        for uid in ("m1", "m2"):
            session.add(Model(uid=uid, download_status=DownloadStatus.downloaded))
        session.flush()
        session.add(
            Artifact(
                model_uid="m1",
                stage=ArtifactStage.rendered,
                status=ArtifactStatus.done,
                key="processed/renders/m1/view_00.png",
            )
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
    predict._loaded.clear()
    return TestClient(api.app)


def _login(client: TestClient, email: str) -> TestClient:
    client.post("/auth/login", json={"email": email, "password": PASSWORD})
    client.headers[CSRF_HEADER] = client.cookies[CSRF_COOKIE]
    return client


@pytest.fixture
def viewer_client(anon_client: TestClient) -> TestClient:
    return _login(anon_client, VIEWER_EMAIL)


def _seed_run(
    status: TrainingStatus = TrainingStatus.completed, weights: str | None = "w.pt"
) -> int:
    with db.session_scope() as session:
        run = TrainingRun(config={}, data_snapshot={}, status=status, weights_uri=weights)
        session.add(run)
        session.flush()
        return run.id


def _stub_inference(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replace the ml package with a recorder, so the route is testable without torch."""
    calls: dict = {}

    class _FakeInfer:
        @staticmethod
        def load_run_model(run_id, storage):
            calls["loads"] = calls.get("loads", 0) + 1
            return object(), None, {}

        @staticmethod
        def classify_model(model, uid, storage):
            calls["uid"] = uid
            return RANKED

    monkeypatch.setattr(predict, "_inference", lambda: _FakeInfer)
    return calls


def test_viewers_can_predict(viewer_client: TestClient, monkeypatch) -> None:
    """A prediction is a statement about the catalog, which any authed user can
    already see — unlike the weights themselves, which stay admin-only (NFR-6)."""
    calls = _stub_inference(monkeypatch)
    run_id = _seed_run()

    body = viewer_client.get("/models/m1/predict").json()

    assert body["run_id"] == run_id
    assert calls["uid"] == "m1"
    assert [row["class_name"] for row in body["predictions"]] == ["chair", "table", "lamp"]
    assert body["predictions"][0]["probability"] == pytest.approx(0.71)


def test_prediction_needs_a_session(anon_client: TestClient) -> None:
    assert anon_client.get("/models/m1/predict").status_code == 401


def test_an_unrendered_model_is_404(viewer_client: TestClient, monkeypatch) -> None:
    """Named precisely rather than surfacing a storage miss: the backends raise
    different exceptions, and "not rendered yet" is a state, not a failure."""
    _stub_inference(monkeypatch)
    _seed_run()

    response = viewer_client.get("/models/m2/predict")

    assert response.status_code == 404
    assert "rendered" in response.json()["detail"]


def test_an_unknown_model_is_404(viewer_client: TestClient, monkeypatch) -> None:
    _stub_inference(monkeypatch)
    _seed_run()

    assert viewer_client.get("/models/nope/predict").status_code == 404


def test_no_trained_run_is_503_not_500(viewer_client: TestClient, monkeypatch) -> None:
    """Nothing is broken — nothing has trained yet. Local dev hits this routinely."""
    _stub_inference(monkeypatch)

    response = viewer_client.get("/models/m1/predict")

    assert response.status_code == 503
    assert "no completed training run" in response.json()["detail"]


def test_a_running_or_weightless_run_does_not_count(viewer_client: TestClient, monkeypatch) -> None:
    _stub_inference(monkeypatch)
    _seed_run(status=TrainingStatus.running)
    _seed_run(status=TrainingStatus.completed, weights=None)
    _seed_run(status=TrainingStatus.failed)

    assert viewer_client.get("/models/m1/predict").status_code == 503


def test_the_newest_trained_run_answers(viewer_client: TestClient, monkeypatch) -> None:
    _stub_inference(monkeypatch)
    _seed_run()
    newest = _seed_run()

    assert viewer_client.get("/models/m1/predict").json()["run_id"] == newest


def test_weights_are_loaded_once_and_reused(viewer_client: TestClient, monkeypatch) -> None:
    """~43MB off object storage per request would dominate the response."""
    calls = _stub_inference(monkeypatch)
    _seed_run()

    viewer_client.get("/models/m1/predict")
    viewer_client.get("/models/m1/predict")

    assert calls["loads"] == 1


def _stub_upload(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Record what classify_upload was handed, without rendering anything.

    The render itself needs a GL context and is covered by the pipeline's own
    tests; what this file is for is the route's contract.
    """
    calls: dict = {}

    def fake(data, file_type, storage):
        calls["bytes"] = len(data)
        calls["file_type"] = file_type
        return 7, RANKED

    monkeypatch.setattr(api, "classify_upload", fake)
    return calls


def test_an_uploaded_mesh_is_classified(viewer_client: TestClient, monkeypatch) -> None:
    """Open to viewers: nothing is stored, so this is not a way past the admin
    gate on FR-9 ingestion."""
    calls = _stub_upload(monkeypatch)

    response = viewer_client.post(
        "/models/predict-upload", files={"file": ("thing.glb", b"mesh-bytes", "model/gltf-binary")}
    )

    assert response.status_code == 200
    assert calls == {"bytes": 10, "file_type": "glb"}
    assert response.json()["run_id"] == 7


def test_ply_is_accepted_even_though_ingest_refuses_it(
    viewer_client: TestClient, monkeypatch
) -> None:
    """The normalized PLY is what a detail page hands out, so refusing it would
    reject the app's own artifact."""
    calls = _stub_upload(monkeypatch)

    response = viewer_client.post(
        "/models/predict-upload",
        files={"file": ("mesh.ply", b"ply-bytes", "application/octet-stream")},
    )

    assert response.status_code == 200
    assert calls["file_type"] == "ply"


def test_an_unsupported_format_is_415(viewer_client: TestClient, monkeypatch) -> None:
    _stub_upload(monkeypatch)

    response = viewer_client.post(
        "/models/predict-upload", files={"file": ("model.fbx", b"x", "application/octet-stream")}
    )

    assert response.status_code == 415


def test_an_unusable_mesh_is_422_not_500(viewer_client: TestClient, monkeypatch) -> None:
    """It parsed as its format and still cannot be rendered — points only, an
    empty scene, zero extent. The user's problem to see."""

    def fake(data, file_type, storage):
        raise api.UnusableMesh("mesh has no faces")

    monkeypatch.setattr(api, "classify_upload", fake)

    response = viewer_client.post(
        "/models/predict-upload", files={"file": ("empty.glb", b"x", "model/gltf-binary")}
    )

    assert response.status_code == 422
    assert "no faces" in response.json()["detail"]


def test_an_empty_file_is_refused(viewer_client: TestClient, monkeypatch) -> None:
    _stub_upload(monkeypatch)

    response = viewer_client.post(
        "/models/predict-upload", files={"file": ("empty.glb", b"", "model/gltf-binary")}
    )

    assert response.status_code == 400


def test_uploading_to_predict_needs_a_session(anon_client: TestClient) -> None:
    response = anon_client.post(
        "/models/predict-upload", files={"file": ("thing.glb", b"x", "model/gltf-binary")}
    )

    # CSRF answers before auth for unsafe methods (server.md#csrf).
    assert response.status_code in (401, 403)


def test_a_render_failure_is_not_blamed_on_the_file(viewer_client: TestClient, monkeypatch) -> None:
    """A broken GL setup is the server's fault. Reporting it as "unusable mesh"
    sent someone looking at their file the first time this ran outside a
    container."""

    def fake(data, file_type, storage):
        raise ValueError("signal only works in main thread of the main interpreter")

    monkeypatch.setattr(api, "classify_upload", fake)

    with pytest.raises(ValueError):
        viewer_client.post(
            "/models/predict-upload", files={"file": ("thing.glb", b"x", "model/gltf-binary")}
        )
