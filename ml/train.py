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
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

from sqlalchemy import select

from app.db import session_scope
from app.models import Label, Model, TrainingMetric, TrainingRun, TrainingStatus


@dataclass
class Config:
    """Hyperparameters for one run. Persisted verbatim to `training_run.config`
    (JSONB), so adding a knob here needs no schema change (config-over-code)."""

    # --- Architecture ---
    # A multi-view CNN: a shared 2D backbone runs on each rendered view, the
    # per-view features are pooled, then a small classifier head maps to the 12
    # classes. Recorded per run so a result is reproducible (NFR-4). A backbone's
    # own layer count is implied by its name (resnet18 = 18 layers) rather than
    # re-listed; head_hidden_dims is the tunable part — the hidden layers of the
    # head and their node counts. No model is built from these yet (the loop is a
    # placeholder); they exist so the config already carries the real shape.
    arch: str = "mvcnn"  # model family
    backbone: str = "resnet18"  # shared per-view 2D CNN (torchvision)
    pretrained: bool = True  # start from ImageNet-pretrained backbone weights
    num_views: int = 12  # rendered views per model (matches the render stage)
    view_pool: str = "max"  # how per-view features combine: "max" | "mean"
    feature_dim: int = 512  # backbone output width fed to the head (resnet18 -> 512)
    # classifier-head hidden layers, one int = nodes in that layer
    head_hidden_dims: list[int] = field(default_factory=lambda: [256])
    dropout: float = 0.5  # dropout in the classifier head
    num_classes: int = 12  # the 12-class roster (ml/taxonomy.py)

    # --- Optimization ---
    epochs: int = 20
    steps_per_epoch: int = 50  # batches per epoch (= len(dataloader) once real)
    log_every: int = 10  # write a loss point every N steps, to throttle DB writes
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


# --- Training loop -----------------------------------------------------------


def run_training(config: Config, run_id: int) -> None:
    """The training loop — deliberately a plain double loop for now.

    There is no model yet: it logs a *synthetic* decaying loss per step and a
    validation loss once per epoch, so the bookkeeping and the dashboard's cost
    curve are exercised end to end with a curve that actually goes down. A real
    multi-view CNN forward/backward pass replaces the synthetic loss here later,
    leaving the ``log_metric`` calls around it unchanged.
    """
    random_source = random.Random(config.seed)  # seeded so the curve reproduces
    total_steps = config.epochs * config.steps_per_epoch
    for epoch in range(config.epochs):
        for step_in_epoch in range(config.steps_per_epoch):
            global_step = epoch * config.steps_per_epoch + step_in_epoch
            # Exponential decay from ~2.5 toward ~0.2 over the run, plus small
            # noise — a stand-in for a real training loss until the model lands.
            train_loss = 0.2 + 2.3 * math.exp(-3.0 * global_step / total_steps)
            train_loss += random_source.uniform(-0.05, 0.05)
            # Validation once per epoch (its last step), sitting a little above the
            # train loss so the train/val gap B4 reads for bias-vs-variance shows.
            is_last_step = step_in_epoch == config.steps_per_epoch - 1
            val_loss = None
            if is_last_step:
                val_loss = train_loss + 0.15 + random_source.uniform(0.0, 0.1)
            # Throttle DB writes: log every `log_every` steps, but always log the
            # epoch's last step so its val_loss point is never dropped.
            if is_last_step or global_step % config.log_every == 0:
                log_metric(run_id, global_step, train_loss, val_loss)
        print(
            f"epoch {epoch + 1}/{config.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
        )


# --- Entry point -------------------------------------------------------------


def main() -> None:
    """Run one training run end to end: snapshot the data, open the run row,
    train, and finalize. Any failure marks the run ``failed`` (so it never
    lingers as ``running``) and re-raises so the traceback is visible.

    Config is the ``Config`` defaults for now — CLI/config-file overrides can be
    added here later without touching the loop or the bookkeeping.
    """
    config = Config()
    snapshot = data_snapshot()
    run_id = create_run(
        config, snapshot, notes=f"{config.arch} baseline, {config.epochs} epochs"
    )
    print(
        f"training_run {run_id}: {snapshot['label_count']} labels, "
        f"{config.epochs} epochs x {config.steps_per_epoch} steps"
    )
    try:
        run_training(config, run_id)
    except Exception:
        finalize_run(run_id, TrainingStatus.failed)
        raise
    finalize_run(run_id, TrainingStatus.completed)
    print(f"training_run {run_id}: completed")


if __name__ == "__main__":
    main()
