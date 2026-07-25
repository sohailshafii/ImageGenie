"""Baseline training loop for the multi-view classifier (M6, ml.md#training).

Deliberately simple, per the M6 complexity budget (CLAUDE.md): a plain loop with a
handful of hyperparameters in `Config`, plus the one non-negotiable — NFR-4
bookkeeping. Every run records its config, the data snapshot it trained on, and
its per-step metrics to the `training_run` / `training_metric` tables, written
directly through a DB session (like the pipeline workers), so the dashboard can
compare runs and the loss curve is answerable.

The model and the multi-view dataset are intentionally minimal placeholders here
(a plain loop over a synthetic loss) — the bookkeeping backbone is the real part
and grows a real model + GCS-render dataset next.

Run via `make train` (which sets PYTHONPATH=server so the DB layer imports).
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from sqlalchemy import select

from app.db import session_scope
from app.models import Label, Model, TrainingMetric, TrainingRun, TrainingStatus


@dataclass
class Config:
    """Hyperparameters for one run. Persisted verbatim to `training_run.config`
    (JSONB), so adding a knob here needs no schema change (config-over-code)."""

    arch: str = "mvcnn"  # model architecture (a label for now; no model yet)
    epochs: int = 20
    steps_per_epoch: int = 50  # batches per epoch (= len(dataloader) once real)
    batch_size: int = 32
    learning_rate: float = 3e-4
    optimizer: str = "adam"  # "adam" | "sgd"
    momentum: float = 0.9  # SGD only
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    seed: int = 0


def data_snapshot() -> dict:
    """Capture *which labels* this run trains on, for reproducibility (NFR-4).

    This is the ``training_run.data_snapshot`` blob. It records the current label
    of every live, labeled model — "current" resolved exactly as the labeling API
    does: the most-recent label per model, so a manual correction wins over the
    weak label (see ``_latest_labels`` in ``app.api``). A content ``label_hash``
    over the sorted (uid, class) pairs makes the training set identifiable: two
    runs with the same hash trained on the same labels, and a changed hash flags
    that the data moved underneath a comparison.

    Only the label set is snapshotted for now — the model and dataset are still
    placeholders. When the real multi-view dataset lands, this narrows to models
    that also have renders in the processed bucket (an added ``filter`` clause).
    """
    with session_scope() as session:
        # DISTINCT ON (model_uid) with created_at/id DESC keeps one row per model,
        # the newest; joined to `model` so soft-deleted models drop out (FR-9).
        current_label_stmt = (
            select(Label.model_uid, Label.class_name)
            .join(Model, Model.uid == Label.model_uid)
            .where(Model.deleted_at.is_(None))
            .distinct(Label.model_uid)
            .order_by(Label.model_uid, Label.created_at.desc(), Label.id.desc())
        )
        # Rows arrive sorted by model_uid (the DISTINCT ON leading key), so the
        # hash below is order-stable without an extra sort.
        labeled_models = session.execute(current_label_stmt).all()

    digest = hashlib.sha256()
    class_to_count: Counter[str] = Counter()
    for model_uid, class_name in labeled_models:
        digest.update(f"{model_uid}\t{class_name}\n".encode())
        class_to_count[class_name] += 1

    return {
        "label_count": len(labeled_models),
        "label_hash": "sha256:" + digest.hexdigest(),
        "as_of": datetime.now(UTC).isoformat(),
        "filter": {"deleted": "excluded", "label": "current (manual over weak)"},
        "class_counts": dict(sorted(class_to_count.items())),
    }


# --- Run bookkeeping (NFR-4) -------------------------------------------------
# Three small helpers own the write path to `training_run` / `training_metric`.
# Each opens its own `session_scope` and commits independently, on purpose: the
# run row lands before the loop starts and each metric lands as it is produced,
# so the dashboard shows a live run with a growing loss curve rather than
# everything appearing at once when training finishes.


def create_run(config: Config, snapshot: dict, notes: str | None = None) -> int:
    """Insert the run row (status defaults to ``running``) and return its id.

    Written before the training loop so the run is visible while it trains; the
    loss curve (``log_metric``) and terminal state (``finalize_run``) attach to
    the returned id.
    """
    with session_scope() as session:
        run = TrainingRun(
            config=asdict(config),  # dataclass -> JSONB blob (config-over-code)
            data_snapshot=snapshot,
            notes=notes,
        )
        session.add(run)
        session.flush()  # assigns run.id from the DB before the scope commits
        return run.id


def log_metric(
    run_id: int, step: int, loss: float, val_loss: float | None = None
) -> None:
    """Append one point to a run's loss curve (``training_metric``), committed on
    its own so the dashboard's cost curve grows while the run is still going.
    ``val_loss`` is null on steps where validation was not evaluated."""
    with session_scope() as session:
        session.add(
            TrainingMetric(run_id=run_id, step=step, loss=loss, val_loss=val_loss)
        )


def finalize_run(
    run_id: int,
    status: TrainingStatus,
    metrics: dict | None = None,
    weights_uri: str | None = None,
) -> None:
    """Mark a run terminal: set its status and ``finished_at``, and optionally the
    dev-set ``metrics`` blob and saved-weights path. Called once on success, or
    with ``status=failed`` from the exception path so a crashed run does not sit
    forever in ``running`` (chunk 5 wires that up)."""
    with session_scope() as session:
        run = session.get(TrainingRun, run_id)
        if run is None:
            raise ValueError(f"training_run {run_id} not found")
        run.status = status
        run.finished_at = datetime.now(UTC)
        if metrics is not None:
            run.metrics = metrics
        if weights_uri is not None:
            run.weights_uri = weights_uri
