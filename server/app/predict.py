"""Classify a model with the most recent trained run (server.md#predicting-a-class).

The API's one *computational* endpoint: everything else reads rows or streams
blobs, while this rebuilds a trained model and runs a forward pass over twelve
rendered views. That difference drives every decision here — which run to use,
how the model is cached, and why the imports are deferred.
"""

from __future__ import annotations

import logging

from sqlalchemy import select

from .artifact_keys import NUM_VIEWS, RAW_SUFFIX_TO_FILE_TYPE
from .db import session_scope
from .models import TrainingRun, TrainingStatus
from .storage import Storage
from .workers.mesh import load_mesh
from .workers.normalize import normalize_mesh
from .workers.render import RESOLUTION, camera_poses, render_views

logger = logging.getLogger(__name__)


class PredictionUnavailable(Exception):
    """No model can answer right now — nothing has trained successfully yet, or
    this deployment has no ml package. A state to report, not a bug: local dev
    hits it routinely, and the UI should say so rather than show an error."""


class UnusableMesh(Exception):
    """The upload parsed as its format but yields nothing renderable — an empty
    scene, points or curves only, zero extent. The user's problem to see (422),
    distinct from a rendering failure, which is the server's (500)."""


# run id -> the loaded model. One entry: a newer run supersedes an older one, and
# holding both would double the resident footprint of a service capped at 2Gi.
#
# A module-level cache is only safe because the API runs at
# `max_instance_count = 1` (infra/api.tf). Raise that and each instance keeps its
# own copy — still correct, just less useful.
_loaded: dict[int, object] = {}


def _inference():
    """Import the ml package lazily, as a module object.

    Deferred rather than imported at module scope for two reasons. Torch costs
    seconds to import, and paying that on every cold start — including for the
    requests that never predict, which is nearly all of them — is a poor trade.
    And the ml package is absent in local dev (`PYTHONPATH=server`), where an
    import-time failure would take down the whole API rather than one route.
    """
    try:
        import infer
    except ImportError as error:  # pragma: no cover - exercised by deployment shape
        raise PredictionUnavailable(
            "this deployment has no ml package, so it cannot classify"
        ) from error
    return infer


def latest_trained_run_id() -> int | None:
    """The newest completed run that actually saved weights.

    Newest rather than best on purpose: "best" would need a metric to rank by,
    and the honest one (a held-out score) exists only for runs someone has
    evaluated. Picking silently by accuracy would also mean the answer changes
    when an evaluation is recorded, which is a surprising thing for a prediction
    to depend on. The run id is returned to the caller either way, so which model
    answered is always visible.
    """
    with session_scope() as session:
        return session.execute(
            select(TrainingRun.id)
            .where(TrainingRun.status == TrainingStatus.completed)
            .where(TrainingRun.weights_uri.is_not(None))
            .order_by(TrainingRun.id.desc())
            .limit(1)
        ).scalar_one_or_none()


def _current_model(storage: Storage):
    """The newest trained run's model, loaded once. Returns `(run_id, model, infer)`."""
    infer = _inference()
    run_id = latest_trained_run_id()
    if run_id is None:
        raise PredictionUnavailable("no completed training run has saved weights yet")

    model = _loaded.get(run_id)
    if model is None:
        # Loading is ~43 MB off object storage plus a state-dict copy, so doing it
        # per request would dominate the response.
        logger.info("loading weights for prediction", extra={"run": run_id})
        model, _config, _snapshot = infer.load_run_model(run_id, storage)
        _loaded.clear()  # keep exactly one run resident
        _loaded[run_id] = model
    return run_id, model, infer


def classify(uid: str, storage: Storage) -> tuple[int, list[tuple[str, float]]]:
    """Classify one rendered model. Returns `(run_id, ranked predictions)`.

    The whole roster comes back ranked, not just the winner — a near-tie between
    `figure` and `animal` is the case worth seeing, and a single label hides it.
    """
    run_id, model, infer = _current_model(storage)
    return run_id, infer.classify_model(model, uid, storage)


# What an uploaded mesh may be. The ingest formats plus PLY, which the pipeline
# produces but never accepts: the normalized PLY is exactly what a user can
# download from a model's detail page, so refusing it here would reject the one
# file the app itself hands out.
PREDICTABLE_TYPES = {**RAW_SUFFIX_TO_FILE_TYPE, ".ply": "ply"}


def classify_upload(
    data: bytes, file_type: str, storage: Storage
) -> tuple[int, list[tuple[str, float]]]:
    """Classify a mesh that is not in the catalog, storing nothing.

    Renders in-request rather than going through the pipeline, which is
    asynchronous and admin-only — neither of which suits "what is this?" from a
    viewer. Nothing is written: no bucket object, no `model` row. The upload is a
    question, not an ingestion, and treating it as one would also hand every
    viewer a way past the admin gate on FR-9 uploads.

    **The mesh is normalized exactly as the pipeline would.** The model was
    trained on origin-centred, unit-scale renders, so skipping that step feeds it
    views it has never seen — and the result would be a confidently wrong
    prediction rather than an error.
    """
    run_id, model, infer = _current_model(storage)

    # Only the mesh-level failures become `UnusableMesh`. Wrapping the render too
    # would report a broken GL setup as a malformed upload — which is exactly what
    # happened the first time this ran outside a container, where pyrender falls
    # back to pyglet and refuses to start off the main thread. That is a server
    # fault and should read as one.
    try:
        mesh = normalize_mesh(load_mesh(data, file_type=file_type))
    except ValueError as error:
        raise UnusableMesh(str(error)) from error

    views = render_views(mesh, camera_poses(NUM_VIEWS), RESOLUTION)
    stacked = infer.views_from_images(views)
    return run_id, infer.classify_views(model, stacked)
