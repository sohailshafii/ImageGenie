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
    TrainingRun,
    TrainingStatus,
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
                "TRUNCATE evaluation, training_metric, training_run, label,"
                " artifact, model, session, app_user RESTART IDENTITY CASCADE"
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


def test_hyperparameters_travel_as_flags(
    admin_client: TestClient, configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def fake_submit(settings, args, display_name):
        captured["args"] = args
        return "job"

    monkeypatch.setattr(api, "submit_training_job", fake_submit)

    response = admin_client.post(
        "/training-runs",
        json={
            "epochs": 6,
            "learning_rate": 0.001,
            "batch_size": 16,
            "optimizer": "sgd",
            "momentum": 0.9,
            "dropout": 0.3,
            "weight_decay": 0.0001,
            "label_smoothing": 0.1,
            "class_weighting": "balanced",
            "notes": "weighted",
        },
    )

    assert response.status_code == 202
    assert captured["args"] == [
        "--device", "cuda", "--num-workers", "4", "--epochs", "6",
        "--learning-rate", "0.001", "--batch-size", "16",
        "--optimizer", "sgd", "--momentum", "0.9", "--dropout", "0.3",
        "--weight-decay", "0.0001", "--label-smoothing", "0.1",
        "--class-weighting", "balanced",
        "--notes", "weighted",
    ]


def test_unset_hyperparameters_contribute_no_flags(
    admin_client: TestClient, configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An omitted knob must leave ml/train.py's Config default alone, not be
    resent as the API's idea of a default."""
    captured = {}

    def fake_submit(settings, args, display_name):
        captured["args"] = args
        return "job"

    monkeypatch.setattr(api, "submit_training_job", fake_submit)

    admin_client.post("/training-runs", json={"epochs": 1, "class_weighting": "balanced"})

    assert captured["args"] == [
        "--device", "cuda", "--num-workers", "4", "--epochs", "1",
        "--class-weighting", "balanced",
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"dropout": 1.0},  # torch requires 0 <= p < 1
        {"dropout": -0.1},
        {"learning_rate": 0},  # not a learning rate at all
        {"learning_rate": -0.1},
        {"batch_size": 0},
        {"optimizer": "rmsprop"},  # ml/train.py builds adam or sgd
        {"class_weighting": "inverse-sqrt"},
        {"label_smoothing": 1.0},
        {"weight_decay": -0.1},
        {"momentum": 1.5},
    ],
)
def test_nonsense_hyperparameters_are_rejected_before_vertex(
    admin_client: TestClient, configured: None, payload: dict
) -> None:
    """These would otherwise crash the container ~15 min in, on a billed GPU."""
    response = admin_client.post("/training-runs", json={"epochs": 1, **payload})

    assert response.status_code == 422


def test_batch_size_that_would_oom_the_gpu_is_rejected(
    admin_client: TestClient, configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure this stops is silent for ~45s and then bills for a dead run.

    Runs 18-20 were one Vertex job retrying itself three times, each dying of
    `torch.OutOfMemoryError` on a batch of 64 — which is 768 images, because the
    12 views ride along. Nothing said no.
    """

    def fail_if_called(settings, args, display_name):  # pragma: no cover
        raise AssertionError("a batch this size must never reach Vertex")

    monkeypatch.setattr(api, "submit_training_job", fail_if_called)

    response = admin_client.post(
        "/training-runs", json={"epochs": 1, "batch_size": training_jobs.MAX_BATCH_SIZE + 1}
    )

    assert response.status_code == 422
    # The multiplier is the part the admin cannot see, so the refusal has to name
    # it — a bare "too large" would leave the 12x still hidden.
    message = response.text
    assert str(training_jobs.VIEWS_PER_MODEL) in message
    assert str((training_jobs.MAX_BATCH_SIZE + 1) * training_jobs.VIEWS_PER_MODEL) in message


def test_the_largest_fitting_batch_is_accepted(
    admin_client: TestClient, configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ceiling itself must launch: an off-by-one here reads as a knob that
    lies about its own range."""
    captured = {}

    def fake_submit(settings, args, display_name):
        captured["args"] = args
        return "job"

    monkeypatch.setattr(api, "submit_training_job", fake_submit)

    response = admin_client.post(
        "/training-runs", json={"epochs": 1, "batch_size": training_jobs.MAX_BATCH_SIZE}
    )

    assert response.status_code == 202
    assert "--batch-size" in captured["args"]


def test_launch_config_reports_the_batch_ceiling(
    admin_client: TestClient, configured: None
) -> None:
    """The form renders the limit and the 12x multiplier from these, so that it
    has no copy of either to drift from."""
    body = admin_client.get("/training-launch").json()

    assert body["views_per_model"] == training_jobs.VIEWS_PER_MODEL
    assert body["max_batch_size"] == training_jobs.MAX_BATCH_SIZE


def test_views_per_model_matches_the_trainer() -> None:
    """`VIEWS_PER_MODEL` is a copy of ml/train.py's `Config.num_views` (the API
    image ships without the ml package, see app/roster.py). If the render stage
    ever produces a different number of views, the copy has to move with it — the
    batch ceiling is derived from it, so drift silently mis-sizes the limit."""
    import train

    assert training_jobs.VIEWS_PER_MODEL == train.Config().num_views


def test_every_emitted_flag_is_one_the_trainer_parses() -> None:
    """The launch route and ml/train.py's argparse live in different files, so a
    `--learning_rate`/`--learning-rate` slip would only surface ~15 min into a
    billed job, when argparse rejects it. Cheaper to assert here."""
    import train

    accepted = {
        option
        for action in train.build_parser()._actions
        for option in action.option_strings
    }

    for flag in api._HYPERPARAMETER_FLAGS.values():
        assert flag in accepted, f"{flag} is not a flag ml/train.py accepts"
    # The run-shape and runtime flags the route hardcodes travel the same path.
    for flag in ("--device", "--num-workers", "--epochs", "--limit", "--notes"):
        assert flag in accepted


# ── Launching an evaluation (M7) ─────────────────────────────────────────────
# Same submit path, same image, different entrypoint. What is new is the state
# these have to check *before* a GPU is billed for 15 minutes to find out.


def _seed_scorable_run(*, weights: bool = True) -> int:
    """A finished run with saved weights — the only kind that can be scored."""
    with db.session_scope() as session:
        run = TrainingRun(
            status=TrainingStatus.completed,
            config={},
            data_snapshot={},
            weights_uri="processed/models/1.pt" if weights else None,
        )
        session.add(run)
        session.flush()
        return run.id


def test_a_viewer_cannot_launch_an_evaluation(anon_client: TestClient) -> None:
    """Scoring spends GPU time, so it is admin-only like training."""
    run_id = _seed_scorable_run()
    viewer = _login(anon_client, VIEWER_EMAIL)

    response = viewer.post(f"/training-runs/{run_id}/evaluations", json={})

    assert response.status_code == 403


def test_evaluating_an_unknown_run_is_404(
    admin_client: TestClient, configured: None
) -> None:
    assert admin_client.post("/training-runs/999/evaluations", json={}).status_code == 404


def test_evaluating_a_run_without_weights_is_refused(
    admin_client: TestClient, configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed or still-training run has nothing to load. Caught here rather
    than ~15 minutes in, when the container finds no checkpoint."""

    def fail_if_called(settings, args, display_name, command=None):  # pragma: no cover
        raise AssertionError("a run with no weights must never reach Vertex")

    monkeypatch.setattr(api, "submit_training_job", fail_if_called)
    run_id = _seed_scorable_run(weights=False)

    response = admin_client.post(f"/training-runs/{run_id}/evaluations", json={})

    assert response.status_code == 409


def test_an_unknown_dev_set_is_refused(
    admin_client: TestClient, configured: None
) -> None:
    run_id = _seed_scorable_run()

    response = admin_client.post(
        f"/training-runs/{run_id}/evaluations", json={"dev_set": "holdout"}
    )

    assert response.status_code == 422


def test_evaluation_runs_the_scorer_not_the_trainer(
    admin_client: TestClient, configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole difference between the two jobs is this command override. Get it
    wrong and the job trains a *new* model on a GPU instead of scoring one."""
    captured = {}

    def fake_submit(settings, args, display_name, command=None):
        captured.update(args=args, display_name=display_name, command=command)
        return "projects/p/locations/l/customJobs/1"

    monkeypatch.setattr(api, "submit_training_job", fake_submit)
    run_id = _seed_scorable_run()

    response = admin_client.post(
        f"/training-runs/{run_id}/evaluations", json={"dev_set": "val"}
    )

    assert response.status_code == 202
    assert captured["command"] == training_jobs.EVALUATE_COMMAND
    assert captured["args"] == [
        "--run", str(run_id), "--dev-set", "val", "--num-workers", "4",
    ]
    assert str(run_id) in captured["display_name"]


def test_the_default_dev_set_is_the_sealed_split(
    admin_client: TestClient, configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`test` is the one number nothing steered against, so it is what an admin
    who expresses no preference gets."""
    captured = {}
    monkeypatch.setattr(
        api,
        "submit_training_job",
        lambda settings, args, display_name, command=None: captured.update(args=args)
        or "job",
    )
    run_id = _seed_scorable_run()

    admin_client.post(f"/training-runs/{run_id}/evaluations", json={})

    assert "--dev-set" in captured["args"]
    assert captured["args"][captured["args"].index("--dev-set") + 1] == "test"


def test_the_command_override_is_absent_for_training() -> None:
    """Training must keep using the image's own ENTRYPOINT — passing a command
    there would be a silent way to run the wrong script on a paid GPU."""
    settings = config.Settings(**CONFIGURED)

    training = training_jobs.build_job_spec(settings, ["--epochs", "1"], "train")
    scoring = training_jobs.build_job_spec(
        settings, ["--run", "1"], "evaluate", training_jobs.EVALUATE_COMMAND
    )

    container = training["jobSpec"]["workerPoolSpecs"][0]["containerSpec"]
    assert "command" not in container
    scoring_container = scoring["jobSpec"]["workerPoolSpecs"][0]["containerSpec"]
    assert scoring_container["command"] == ["python", "evaluate.py"]


def test_the_offered_dev_sets_are_ones_the_scorer_accepts() -> None:
    """The API's copy and ml/evaluate.py's argparse are in different packages, so
    a dev set offered here but unknown there fails ~15 minutes into a paid job."""
    import evaluate

    accepted = evaluate.build_parser().parse_args(["--run", "1"])
    assert accepted.dev_set == "test"  # the same default the API applies
    for dev_set in training_jobs.EVALUATION_DEV_SETS:
        assert dev_set in evaluate.DEV_SETS, f"{dev_set} is not one evaluate.py scores"
    # The partitions, exactly. `lvis` is absent only because its CSV is not in the
    # training image; when that changes this assertion is the thing to update, and
    # until then it stops the dev set being offered before it can work.
    assert training_jobs.EVALUATION_DEV_SETS == evaluate.PARTITIONS
