"""Submit a Vertex AI custom training job (web.md#starting-a-training-run).

The launch half of the "start a training run" page. Creating a custom job is a
single authenticated POST, so this uses `httpx` and ambient credentials rather
than pulling `google-cloud-aiplatform` — a large dependency — into the API image
for one call. Same reasoning as `app/mail.py` using httpx for Resend.

**This mirrors `ml/vertex_job.yaml`**, which `make train-cloud` submits through
gcloud. Two callers with nothing in common (an operator's shell, an HTTP request)
and no shared runtime — the API image does not contain `ml/` — so the spec exists
twice. Change one, change the other: machine shape, spot strategy, env, and the
service account all have to match or a UI-launched run behaves differently from a
CLI-launched one.
"""

from __future__ import annotations

import logging

import httpx

from .config import Settings

logger = logging.getLogger(__name__)

# Matches ml/vertex_job.yaml. n1-standard-8 is the smallest shape a T4 attaches
# to; T4 is the cheapest widely-available spot GPU (server.md#training-gpu).
MACHINE_TYPE = "n1-standard-8"
ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"

# Vertex rejects a job whose creation takes too long to authenticate; this is a
# submit call, not the training itself, so it should be quick or fail.
SUBMIT_TIMEOUT_SECONDS = 30.0

# ── How large a batch the machine above can actually hold ───────────────────
#
# `batch_size` counts *models*, but a model is 12 rendered views and
# `ml/model.py`'s forward folds them into the batch (`views.flatten(0, 1)`), so
# the shared backbone sees `batch_size * VIEWS_PER_MODEL` images per pass. The
# multiplier is invisible at the point where the number is typed, which is how
# runs 18-20 came to die of `torch.OutOfMemoryError` ~45s in, on a billed GPU.
#
# Duplicated from `ml/train.py`'s `Config.num_views` for the same reason as
# `app/roster.py`: the API image ships `server/app` and the built SPA and nothing
# else, so importing the ml package here would crash the service in production
# while working in every local test. `tests/test_roster.py` asserts it still
# matches.
VIEWS_PER_MODEL = 12

# Measured on the T4 above, not estimated. 384 images per pass (batch 32) trained
# fine; 768 (batch 64) died 148 MiB past the card's 14.58 GiB. The ceiling sits
# below the failure rather than at it, because the margin an allocator has left
# after fragmentation is not something a form can know.
MAX_IMAGES_PER_FORWARD = 512

# What the admin actually types, in models.
MAX_BATCH_SIZE = MAX_IMAGES_PER_FORWARD // VIEWS_PER_MODEL

# ── Scoring a finished run ──────────────────────────────────────────────────
#
# The same image, a different entrypoint. `ml/Dockerfile` does `COPY ml/*.py`, so
# evaluate.py and everything it imports are already in there next to train.py —
# the only thing standing between them is the ENTRYPOINT, which this overrides.
# A second image would double the build time and the Artifact Registry footprint
# to ship a copy of files that are already present.
EVALUATE_COMMAND = ["python", "evaluate.py"]

# The dev sets a run can be scored against, duplicated from ml/evaluate.py's
# `DEV_SETS` for the same reason as app/roster.py — the API image ships without
# the ml package. `tests/test_training_launch.py` asserts the copy still matches.
#
# `lvis` is the second dev set (FR-7) and behaves unlike the other three: it is a
# separate corpus rather than a slice of ours, and its selection lives in a file
# rather than the database. A job reads that file from the processed bucket
# (`make devset-push` puts it there), so **an evaluation asked for `lvis` fails if
# nobody has pushed it** — with a message naming the push, not a stack trace.
EVALUATION_DEV_SETS: tuple[str, ...] = ("test", "val", "train", "lvis")


class TrainingLaunchError(RuntimeError):
    """Vertex refused the job, or the API is not configured to submit one."""


def launch_configured(settings: Settings) -> bool:
    """True when this deployment can actually submit a job.

    Local dev has no Vertex, so the route reports that plainly instead of
    failing at the POST with something inscrutable.
    """
    return bool(
        settings.train_image and settings.trainer_service_account and settings.vertex_project
    )


def _access_token() -> str:
    """An OAuth token for the API's own service account (ambient on Cloud Run)."""
    import google.auth
    from google.auth.transport.requests import Request as AuthRequest

    credentials, _project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    if not credentials.valid:
        credentials.refresh(AuthRequest())
    return credentials.token


def build_job_spec(
    settings: Settings,
    args: list[str],
    display_name: str,
    command: list[str] | None = None,
) -> dict:
    """The CustomJob body. Split out from the POST so tests can assert on it.

    `command` overrides the image's ENTRYPOINT, which is `python train.py`. The
    training image already carries every ml module — `COPY ml/*.py` — so scoring a
    run needs no second image, only a different entrypoint. Left unset for
    training, where the image's own default is the right one.
    """
    container = {
        "imageUri": settings.train_image,
        "args": args,
    }
    if command is not None:
        container["command"] = command
    return {
        "displayName": display_name,
        "jobSpec": {
            "workerPoolSpecs": [
                {
                    "machineSpec": {
                        "machineType": MACHINE_TYPE,
                        "acceleratorType": ACCELERATOR_TYPE,
                        "acceleratorCount": 1,
                    },
                    "replicaCount": 1,
                    "containerSpec": {
                        **container,
                        "env": [
                            # Vertex has no Cloud SQL socket and a dynamic egress
                            # IP, so the job dials through the connector over IAM.
                            {
                                "name": "IMAGEGENIE_CLOUDSQL_INSTANCE",
                                "value": (
                                    f"{settings.vertex_project}:{settings.vertex_region}"
                                    ":imagegenie-pg"
                                ),
                            },
                            # A secret *name*: everything in this env block is
                            # visible in the job's metadata, and the URL carries
                            # the database password.
                            {
                                "name": "IMAGEGENIE_DATABASE_URL_SECRET",
                                "value": (
                                    f"projects/{settings.vertex_project}/secrets/"
                                    "imagegenie-database-url/versions/latest"
                                ),
                            },
                        ],
                    },
                }
            ],
            # Spot is what keeps training inside NFR-1's $5–20. A preemption costs
            # the run, not the work: train.py checkpoints after every epoch.
            "scheduling": {"strategy": "SPOT"},
            "serviceAccount": settings.trainer_service_account,
        },
    }


def submit_training_job(
    settings: Settings,
    args: list[str],
    display_name: str,
    command: list[str] | None = None,
) -> str:
    """Create the job and return its Vertex resource name.

    Raises `TrainingLaunchError` on a refusal, with Vertex's own message — a
    quota denial or a missing IAM binding is something the admin needs to read,
    not a generic 500.
    """
    if not launch_configured(settings):
        raise TrainingLaunchError(
            "this deployment is not configured to launch training jobs "
            "(IMAGEGENIE_TRAIN_IMAGE / TRAINER_SERVICE_ACCOUNT / VERTEX_PROJECT)"
        )

    endpoint = (
        f"https://{settings.vertex_region}-aiplatform.googleapis.com/v1/"
        f"projects/{settings.vertex_project}/locations/{settings.vertex_region}/customJobs"
    )
    try:
        response = httpx.post(
            endpoint,
            json=build_job_spec(settings, args, display_name, command),
            headers={"Authorization": f"Bearer {_access_token()}"},
            timeout=SUBMIT_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as error:  # network, DNS, timeout
        raise TrainingLaunchError(f"could not reach Vertex AI: {error}") from error

    if response.status_code >= 400:
        # Vertex's message names the actual cause (quota, IAM, bad image), which
        # is far more useful to an admin than the status code.
        detail = response.text[:500]
        logger.warning("vertex rejected the job: %s %s", response.status_code, detail)
        raise TrainingLaunchError(f"Vertex AI rejected the job: {detail}")

    return response.json().get("name", "")
