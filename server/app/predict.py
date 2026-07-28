"""Classify a model with the most recent trained run (server.md#predicting-a-class).

The API's one *computational* endpoint: everything else reads rows or streams
blobs, while this rebuilds a trained model and runs a forward pass over twelve
rendered views. That difference drives every decision here — which run to use,
how the model is cached, and why the imports are deferred.
"""

from __future__ import annotations

import logging

from sqlalchemy import select

from .db import session_scope
from .models import TrainingRun, TrainingStatus
from .storage import Storage

logger = logging.getLogger(__name__)


class PredictionUnavailable(Exception):
    """No model can answer right now — nothing has trained successfully yet, or
    this deployment has no ml package. A state to report, not a bug: local dev
    hits it routinely, and the UI should say so rather than show an error."""


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


def classify(uid: str, storage: Storage) -> tuple[int, list[tuple[str, float]]]:
    """Classify one rendered model. Returns `(run_id, ranked predictions)`.

    The whole roster comes back ranked, not just the winner — a near-tie between
    `figure` and `animal` is the case worth seeing, and a single label hides it.
    """
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

    return run_id, infer.classify_model(model, uid, storage)
