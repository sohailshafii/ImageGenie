"""ORM entities for the metadata DB (server.md#database).

The DB is the source of truth for pipeline state; object storage holds the heavy
blobs and the DB stores only their keys. Milestone 2 defines ``model`` (the
download stage's state); milestone 4 adds ``artifact`` (the convert / normalize /
render outputs); ``label`` / ``training_run`` / ``user`` land with their stages.
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import ARRAY, DateTime, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class DownloadStatus(str, enum.Enum):
    """Lifecycle of a model's raw-mesh download."""

    pending = "pending"
    downloaded = "downloaded"
    failed = "failed"


class Model(Base):
    """A source 3D model and its download state — one row per store object.

    ``uid`` is the store (Objaverse) id and the primary key, so the download
    worker's ``INSERT ... ON CONFLICT (uid)`` upsert is race-safe under
    at-least-once redelivery (NFR-2).
    """

    __tablename__ = "model"

    uid: Mapped[str] = mapped_column(primary_key=True)
    source_url: Mapped[str | None] = mapped_column(default=None)
    license: Mapped[str | None] = mapped_column(default=None)
    # Store metadata, backfilled by `app.backfill_metadata` (the download worker
    # doesn't persist annotations). Shown in the labeling UI to aid the decision
    # — on the ambiguous classes the title is often what settles it. Nullable
    # because a model can be ingested long before its metadata is fetched.
    title: Mapped[str | None] = mapped_column(default=None)
    tags: Mapped[list[str] | None] = mapped_column(ARRAY(String), default=None)
    download_status: Mapped[DownloadStatus] = mapped_column(default=DownloadStatus.pending)
    content_hash: Mapped[str | None] = mapped_column(default=None)
    raw_key: Mapped[str | None] = mapped_column(default=None)
    # Soft delete (FR-9, admin-only). Set to hide a model from the labeling UI
    # without dropping its rows or blobs, so a mistaken delete is recoverable and
    # `app.reconcile_from_storage` — which rebuilds from storage — doesn't
    # resurrect it (the upsert leaves this column alone). Null = live.
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )


class ArtifactStage(str, enum.Enum):
    """The preprocessing stage that produced an artifact (server.md#queue)."""

    converted = "converted"
    normalized = "normalized"
    rendered = "rendered"


class ArtifactStatus(str, enum.Enum):
    """Lifecycle of a single stage's output for one model."""

    pending = "pending"
    done = "done"
    failed = "failed"


class Artifact(Base):
    """One preprocessing stage's output for one model (server.md#database).

    The DB stores only the object ``key`` (single-file stages) or key **prefix**
    (the render stage's per-view PNGs), never the blob itself. The unique
    ``(model_uid, stage)`` constraint backs the workers' ``INSERT ... ON CONFLICT``
    upsert, so at-least-once redelivery never duplicates a stage's row (NFR-2).
    """

    __tablename__ = "artifact"
    __table_args__ = (
        UniqueConstraint("model_uid", "stage", name="uq_artifact_model_stage"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_uid: Mapped[str] = mapped_column(ForeignKey("model.uid"), index=True)
    stage: Mapped[ArtifactStage] = mapped_column()
    key: Mapped[str] = mapped_column()
    content_hash: Mapped[str | None] = mapped_column(default=None)
    status: Mapped[ArtifactStatus] = mapped_column(default=ArtifactStatus.pending)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )


class PipelineStage(str, enum.Enum):
    """A processing stage a job can fail in — includes download, unlike
    ``ArtifactStage``, which only covers stages that produce an artifact."""

    download = "download"
    convert = "convert"
    normalize = "normalize"
    render = "render"


class DeadLetter(Base):
    """A job that failed a stage, with the error that caused it.

    Recorded by the worker at nack time, because **that is the only place the
    error text exists** — a Pub/Sub dead-letter message carries the original
    payload and a delivery count, never the reason. Keeping it here also means
    the admin view is a plain DB read rather than a destructive pull from the
    DLQ subscription, and that failures outlive Pub/Sub's 7-day retention.

    Unique on ``(model_uid, stage)``: a retried job that fails again updates the
    row rather than adding another, so the list shows current state and not a
    log of every attempt.
    """

    __tablename__ = "dead_letter"
    __table_args__ = (
        UniqueConstraint("model_uid", "stage", name="uq_dead_letter_model_stage"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_uid: Mapped[str] = mapped_column(index=True)  # no FK: a download can
    # fail before the model row exists, and losing the record would be worse
    stage: Mapped[PipelineStage] = mapped_column()
    error: Mapped[str] = mapped_column()
    # Pub/Sub's count for this message; at the subscription's max it stops being
    # redelivered and goes to the dead-letter topic.
    delivery_attempt: Mapped[int | None] = mapped_column(default=None)
    failed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    # Set when an admin re-enqueues it; the row is kept so the history is visible.
    replayed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )


class LabelSource(str, enum.Enum):
    """Where a label came from — the weak-labeling rules or a human correction."""

    weak = "weak"
    manual = "manual"


class Label(Base):
    """A class label for a model (server.md#database, ml.md#weak-label-policy).

    Weak (rule-derived) and manual (human-corrected via the labeling UI) labels are
    kept as **distinct rows** so weak-vs-corrected analysis stays possible — the
    frontend's "current" label for a model is its most recent one.
    """

    __tablename__ = "label"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_uid: Mapped[str] = mapped_column(ForeignKey("model.uid"), index=True)
    class_name: Mapped[str] = mapped_column()  # one of the 12-class roster (ml/taxonomy.py)
    source: Mapped[LabelSource] = mapped_column()
    confidence: Mapped[float | None] = mapped_column(default=None)  # weak labels only
    annotator: Mapped[str | None] = mapped_column(default=None)  # user email, for manual
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class TrainingStatus(str, enum.Enum):
    """Lifecycle of a training run."""

    running = "running"
    completed = "completed"
    failed = "failed"


class TrainingRun(Base):
    """One model-training run and its bookkeeping (FR-6, NFR-4).

    Reproducibility rests on three JSON blobs that mirror NFR-4's three pillars:
    ``config`` (the hyperparameters), ``data_snapshot`` (which labels the run
    trained on), and ``metrics`` (dev-set evaluation). They are blobs rather than
    typed columns so a new hyperparameter or metric needs no migration
    (config-over-code — CLAUDE.md's M6 budget). The per-step loss curve lives in
    ``training_metric``. The training script writes these rows directly through a
    DB session, like the pipeline workers — the API only *reads* them for the
    dashboard, so no write endpoints exist.
    """

    __tablename__ = "training_run"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    status: Mapped[TrainingStatus] = mapped_column(default=TrainingStatus.running)
    # Hyperparameters: learning rate, momentum, Adam betas/eps, batch size, epochs,
    # architecture, etc. A blob so adding a knob needs no schema change.
    config: Mapped[dict] = mapped_column(JSONB)
    # Which data the run trained on — count, content hash, as-of time, any filter —
    # so a result is reproducible (NFR-4). A blob for the same reason as config.
    data_snapshot: Mapped[dict] = mapped_column(JSONB)
    # Dev-set evaluation: per-set accuracy, per-class precision/recall, confusion
    # matrix. Null until the run is evaluated (bias/variance analysis, FR-7).
    metrics: Mapped[dict | None] = mapped_column(JSONB, default=None)
    # GCS path to the saved weights; set when the run completes.
    weights_uri: Mapped[str | None] = mapped_column(default=None)
    # Free-text description for the dashboard, e.g. "mvcnn baseline, 20 epochs".
    notes: Mapped[str | None] = mapped_column(default=None)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )


class TrainingMetric(Base):
    """One point on a run's loss curve — a row per logged step (B2).

    ``val_loss`` and ``val_accuracy`` are nullable because validation is typically
    evaluated less often than the training loss (e.g. once per epoch), so steps
    without it store null. The gap between ``loss`` and ``val_loss`` is what
    reveals variance (overfitting).

    ``val_accuracy`` is stored alongside the losses rather than derived from them:
    it is the number a human actually judges a run by, and on this corpus it is
    the one that exposes a failure loss hides. The classes are skewed ~7.7:1
    (`weapon` alone is ~18%), so a model that collapses onto the majority class
    still posts a falling loss while being useless — the accuracy sitting flat at
    the majority-class rate is what makes that visible.
    """

    __tablename__ = "training_metric"

    run_id: Mapped[int] = mapped_column(ForeignKey("training_run.id"), primary_key=True)
    step: Mapped[int] = mapped_column(primary_key=True)  # epoch or step index
    loss: Mapped[float] = mapped_column()  # training loss at this step
    val_loss: Mapped[float | None] = mapped_column(default=None)  # validation loss, if computed
    # Top-1 accuracy on the val split, 0..1. Null on steps without validation.
    val_accuracy: Mapped[float | None] = mapped_column(default=None)


class Evaluation(Base):
    """One run scored against one dev set (FR-7, M7).

    A table rather than another blob on ``training_run`` because FR-7 asks for
    **two** dev sets and the roster of them is expected to grow: the held-out
    ``test`` split first, an independently-annotated set (e.g. the LVIS gold set)
    after. A column per dev set would need a migration each time; a row per
    (run, dev set) needs none, and it lets one run be re-scored later without
    overwriting what it scored before.

    Distinct from ``training_run.metrics``, which is the run's own end-of-training
    report on ``val``. That one is written by the trainer as part of the run; these
    are written afterwards, by ml/evaluate.py, against data the run never saw.

    The row is written **when scoring starts**, not when it finishes, so an
    evaluation in flight is visible rather than being nothing at all until minutes
    later — and one that dies says so instead of never arriving. That is the same
    bargain ``training_run`` makes, and the reason ``report`` is nullable.
    """

    __tablename__ = "evaluation"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("training_run.id"), index=True)
    # Which dev set was scored: "test" today, e.g. "lvis_gold" later. A string
    # rather than an enum so adding a dev set stays a code-only change.
    dev_set: Mapped[str] = mapped_column()
    # Deliberately the *same* enum as TrainingRun, not a parallel copy: the
    # lifecycle is identical (started, finished, died), the dashboard renders both
    # with one badge, and a second Postgres enum type carrying the same three
    # values would be duplication the database then has to be migrated to keep in
    # step. If the two lifecycles ever diverge, that is the moment to split them.
    status: Mapped[TrainingStatus] = mapped_column(default=TrainingStatus.running)
    # Same shape as training_run.metrics (ml/metrics.py) so the two are directly
    # comparable and the frontend renders both with one component. Null while the
    # evaluation is still running, and on one that failed.
    report: Mapped[dict | None] = mapped_column(JSONB, default=None)
    # Why it failed, for a `failed` row. The alternative to storing it is reading
    # Vertex's log stream, which means leaving the app to find out why something
    # in the app did not work.
    error: Mapped[str | None] = mapped_column(default=None)
    # The label_hash of the trainable set *at scoring time*. The split is
    # recomputed from the live DB rather than stored, so a label added since the
    # run shifts the partition and quietly changes what "test" means. Recording
    # the hash here is what lets a later check say which reports were scored
    # against the run's own split and which were not — unrecoverable if skipped.
    label_hash: Mapped[str | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class UserRole(str, enum.Enum):
    """Access role — ``user`` can view, ``admin`` can also correct + upload (FR-8)."""

    user = "user"
    admin = "admin"


class User(Base):
    """An authenticated account (server.md#database, web.md#auth--roles).

    Table is ``app_user`` — ``user`` is a reserved word in PostgreSQL.
    """

    __tablename__ = "app_user"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(unique=True, index=True)
    role: Mapped[UserRole] = mapped_column(default=UserRole.user)
    password_hash: Mapped[str] = mapped_column()
    verified: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Invite(Base):
    """An admin-minted, email-bound signup invitation (web.md#auth--roles).

    Signup is invite-only, so this table is the gate on account creation. Keyed by
    ``email`` so re-inviting the same address refreshes the existing invite rather
    than accumulating rows — matching the frontend's idempotent-per-email contract.
    """

    __tablename__ = "invite"

    email: Mapped[str] = mapped_column(primary_key=True)  # normalized lowercase
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    accepted: Mapped[bool] = mapped_column(default=False)
    invited_by: Mapped[str | None] = mapped_column(default=None)  # admin's email
    # The role the account is created with when this invite is accepted — the
    # admin chooses viewer (``user``) or ``admin`` at invite time (FR-8).
    role: Mapped[UserRole] = mapped_column(default=UserRole.user)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class EmailVerification(Base):
    """A one-time email-verification token (web.md#auth--roles).

    Stored as a **SHA-256 hash**, never in the clear: a token grants the right to
    verify an account, so a leaked DB snapshot (or a stray log of a query result)
    shouldn't hand that out. Plain SHA-256 rather than bcrypt is right here —
    these are 256-bit random values, not guessable secrets, so there is nothing
    for a slow hash to defend against.
    """

    __tablename__ = "email_verification"

    token_hash: Mapped[str] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("app_user.id"), index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class LoginSession(Base):
    """A login session — the opaque token held in an httpOnly cookie maps here.

    Server-side sessions (not a stateless JWT) so logout can revoke immediately
    (web.md#auth--roles). ``expires_at`` is set by the app at creation. Named
    ``LoginSession`` to avoid colliding with SQLAlchemy's ``Session``.
    """

    __tablename__ = "session"

    token: Mapped[str] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("app_user.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    # timezone-aware so the app's aware expiry round-trips (naive would mismatch
    # datetime.now(timezone.utc) at comparison time).
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
