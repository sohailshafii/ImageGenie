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

from dataclasses import dataclass


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
