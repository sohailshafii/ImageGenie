"""Launching a training run from the UI (web.md#starting-a-training-run).

This is the API's first training *write* route, and the first request that spends
money on a GPU — so the cases that matter are the ones where it must not: a
non-admin, a deployment with no Vertex, and a refusal from Vertex itself.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text

from app import api, config, db, training_jobs
from app.artifact_keys import renders_prefix
from app.models import (
    Artifact,
    ArtifactStage,
    ArtifactStatus,
    DownloadStatus,
    Label,
    LabelSource,
    Model,
    User,
    UserRole,
)
from app.security import CSRF_COOKIE, CSRF_HEADER, hash_password

ADMIN_EMAIL = "admin@imagegenie.dev"
VIEWER_EMAIL = "viewer@imagegenie.dev"
PASSWORD = "genie-secret"

CONFIGURED = {
    "train_image": "us-central1-docker.pkg.dev/proj/imagegenie/train:abc1234",
    "trainer_service_account": "imagegenie-trainer@proj.iam.gserviceaccount.com",
    "vertex_project": "proj",
    "vertex_region": "us-central1",
}


@pytest.fixture
def anon_client(pg_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(db, "get_engine", lambda: pg_engine)
    with pg_engine.begin() as connection:
        connection.execute(
            text(
                "TRUNCATE training_metric, training_run, label, artifact, model,"
                " session, app_user RESTART IDENTITY CASCADE"
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


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "get_settings", lambda: config.Settings(**CONFIGURED))


def _seed_trainable(uid: str, *, rendered: bool = True, labeled: bool = True) -> None:
    with db.session_scope() as session:
        session.add(Model(uid=uid, download_status=DownloadStatus.downloaded))
        session.flush()
        if labeled:
            session.add(Label(model_uid=uid, class_name="chair", source=LabelSource.weak))
        if rendered:
            session.add(
                Artifact(
                    model_uid=uid,
                    stage=ArtifactStage.rendered,
                    status=ArtifactStatus.done,
                    key=renders_prefix(uid),
                )
            )


# ── Access ──────────────────────────────────────────────────────────────────


def test_a_viewer_cannot_launch(anon_client: TestClient) -> None:
    """Spending money is admin-only."""
    viewer = _login(anon_client, VIEWER_EMAIL)
    assert viewer.post("/training-runs", json={"epochs": 1}).status_code == 403
    assert viewer.get("/training-launch").status_code == 403


def test_anonymous_cannot_launch(anon_client: TestClient) -> None:
    # CSRF answers before auth on an unsafe method, so this is 403 not 401.
    assert anon_client.post("/training-runs", json={"epochs": 1}).status_code in (401, 403)
    assert anon_client.get("/training-launch").status_code == 401


# ── The launch form's inputs ────────────────────────────────────────────────


def test_launch_config_reports_the_trainable_count(
    admin_client: TestClient, configured: None
) -> None:
    """The form puts "N of M" in front of the admin, so M has to be real."""
    _seed_trainable("a")
    _seed_trainable("b")
    _seed_trainable("unrendered", rendered=False)  # labeled but not rendered
    _seed_trainable("unlabeled", labeled=False)

    body = admin_client.get("/training-launch").json()

    assert body["trainable_count"] == 2  # only the labeled AND rendered ones
    assert body["configured"] is True
    assert body["image"].endswith(":abc1234")  # the exact tag that will run


def test_launch_config_says_so_when_unconfigured(admin_client: TestClient) -> None:
    """Local dev has no Vertex; the form disables the button on this rather than
    letting the admin click into a 5xx."""
    body = admin_client.get("/training-launch").json()

    assert body["configured"] is False
    assert body["image"] is None


# ── Launching ───────────────────────────────────────────────────────────────


def test_launch_submits_the_requested_arguments(
    admin_client: TestClient, configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def fake_submit(settings, args, display_name):
        captured["args"] = args
        captured["display_name"] = display_name
        return "projects/1/locations/us-central1/customJobs/42"

    monkeypatch.setattr(api, "submit_training_job", fake_submit)

    response = admin_client.post(
        "/training-runs", json={"epochs": 3, "limit": 500, "notes": "smoke"}
    )

    assert response.status_code == 202  # accepted; the run row appears later
    assert captured["args"] == [
        "--device", "cuda", "--num-workers", "4",
        "--epochs", "3", "--limit", "500", "--notes", "smoke",
    ]
    assert response.json()["job_name"].endswith("/42")


def test_no_limit_means_the_whole_trainable_set(
    admin_client: TestClient, configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def fake_submit(settings, args, display_name):
        captured["args"] = args
        return "job"

    monkeypatch.setattr(api, "submit_training_job", fake_submit)

    admin_client.post("/training-runs", json={"epochs": 1})

    assert "--limit" not in captured["args"]


def test_launching_without_vertex_is_503_not_500(admin_client: TestClient) -> None:
    """Nothing is broken — this deployment just has nowhere to submit."""
    response = admin_client.post("/training-runs", json={"epochs": 1})

    assert response.status_code == 503


def test_a_vertex_refusal_surfaces_its_own_message(
    admin_client: TestClient, configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Quota, IAM and bad-image failures all arrive this way, and the text is
    what tells the admin which — so it must not be swallowed into a generic 500."""

    def refuse(settings, args, display_name):
        raise training_jobs.TrainingLaunchError("Vertex AI rejected the job: quota exceeded")

    monkeypatch.setattr(api, "submit_training_job", refuse)

    response = admin_client.post("/training-runs", json={"epochs": 1})

    assert response.status_code == 502
    assert "quota exceeded" in response.json()["detail"]


@pytest.mark.parametrize("payload", [{"epochs": 0}, {"epochs": 1, "limit": 0}])
def test_nonsense_sizes_are_rejected(
    admin_client: TestClient, configured: None, payload: dict
) -> None:
    assert admin_client.post("/training-runs", json=payload).status_code == 422


# ── The job spec ────────────────────────────────────────────────────────────


def test_job_spec_matches_the_cli_path() -> None:
    """It must agree with ml/vertex_job.yaml — the spec lives in two places
    because the API image has no `ml/`, so drift is silent and would mean a
    UI-launched run behaving differently from a CLI-launched one."""
    settings = config.Settings(**CONFIGURED)

    spec = training_jobs.build_job_spec(settings, ["--epochs", "1"], "imagegenie-train")

    pool = spec["jobSpec"]["workerPoolSpecs"][0]
    assert pool["machineSpec"]["machineType"] == "n1-standard-8"
    assert pool["machineSpec"]["acceleratorType"] == "NVIDIA_TESLA_T4"
    assert spec["jobSpec"]["scheduling"]["strategy"] == "SPOT"  # the budget rests on this
    assert spec["jobSpec"]["serviceAccount"].startswith("imagegenie-trainer@")
    assert pool["containerSpec"]["imageUri"].endswith(":abc1234")

    env = {entry["name"]: entry["value"] for entry in pool["containerSpec"]["env"]}
    assert env["IMAGEGENIE_CLOUDSQL_INSTANCE"] == "proj:us-central1:imagegenie-pg"
    # A secret *name*, never the URL: a job's env is visible in its metadata.
    assert env["IMAGEGENIE_DATABASE_URL_SECRET"].startswith("projects/proj/secrets/")
    assert "postgresql" not in str(env)


def test_launch_configured_needs_every_setting() -> None:
    assert training_jobs.launch_configured(config.Settings(**CONFIGURED)) is True
    for missing in ("train_image", "trainer_service_account", "vertex_project"):
        partial = {key: value for key, value in CONFIGURED.items() if key != missing}
        assert training_jobs.launch_configured(config.Settings(**partial)) is False
