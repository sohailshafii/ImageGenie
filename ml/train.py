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
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select

from app.db import session_scope
from app.models import Label, Model


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
