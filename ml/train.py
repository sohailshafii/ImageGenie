"""Training loop for the multi-view classifier (M6, ml.md#training).

Deliberately simple, per the M6 complexity budget (CLAUDE.md): a plain loop with a
handful of hyperparameters in `Config`, plus the one non-negotiable — NFR-4
bookkeeping. Every run records its config, the data snapshot it trained on, and
its per-step metrics to the `training_run` / `training_metric` tables, written
directly through a DB session (like the pipeline workers), so the dashboard can
compare runs and the loss curve is answerable.

The model (ml/model.py) is a multi-view CNN over the rendered views, which the
dataset (ml/dataset.py) streams from the processed bucket; the trainable set is
the models that are both labeled and rendered, split per class (ml/splits.py).

Run via `make train` (which sets PYTHONPATH=server so the DB layer imports). The
device defaults to CPU (the local-first path); the cloud config sets "cuda".
"""

from __future__ import annotations

import argparse
import hashlib
import io
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

import torch
from dataset import MultiViewDataset
from metrics import evaluation_report
from model import MultiViewCNN
from splits import DatasetSplit, split_sizes, stratified_split
from sqlalchemy import select
from taxonomy import ROSTER
from torch import nn
from torch.utils.data import DataLoader

from app.artifact_keys import weights_key
from app.config import get_settings
from app.db import session_scope
from app.models import (
    Artifact,
    ArtifactStage,
    ArtifactStatus,
    Label,
    Model,
    TrainingMetric,
    TrainingRun,
    TrainingStatus,
)
from app.storage import Storage, build_storage


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
    # head and their node counts.
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
    log_every: int = 10  # write a loss point every N steps, to throttle DB writes
    batch_size: int = 32
    learning_rate: float = 3e-4
    optimizer: str = "adam"  # "adam" | "sgd"
    momentum: float = 0.9  # SGD only
    # --- Regularization ---
    # All three default to off, so an unset knob reproduces the previous
    # behaviour exactly and a run that sets one can be compared against a run
    # that didn't (the resolved value is recorded either way, NFR-4).
    weight_decay: float = 0.0  # L2 penalty on the weights; 0 disables it
    label_smoothing: float = 0.0  # softens the CE target; helps with noisy labels
    # How much each class contributes to the loss. "none" = every sample counts
    # the same, so a 7.7:1 class skew pulls the model toward the head classes and
    # the tail collapses (the val macro-recall vs accuracy gap). "balanced"
    # weights each class inversely to its frequency in the *training* split, so
    # one rare-class mistake costs as much as several common-class ones.
    class_weighting: str = "none"  # "none" | "balanced"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    seed: int = 0

    # --- Runtime ---
    # "cpu" is the default so the local smoke never depends on a GPU (the locked
    # local-first decision); the cloud config sets "cuda". "auto" (cuda>mps>cpu)
    # and "mps" are also accepted.
    device: str = "cpu"
    # DataLoader worker processes. 0 = load in the main process, which sidesteps
    # having to pickle the storage client; raise it once at real scale.
    num_workers: int = 0


def load_trainable_samples() -> list[tuple[str, str]]:
    """The trainable set: the current label of every live, *rendered* model.

    A model must be labeled **and** have finished rendering (its 12 views exist)
    to be usable, so both are required: the join to `artifact` drops models the
    pipeline hasn't rendered, the join to `model` + `deleted_at` drops soft-deleted
    ones, and DISTINCT ON keeps the newest label (manual over weak), matching the
    labeling API's resolution.
    """
    with session_scope() as session:
        stmt = (
            select(Label.model_uid, Label.class_name)
            .join(Model, Model.uid == Label.model_uid)
            .join(Artifact, Artifact.model_uid == Label.model_uid)
            .where(Model.deleted_at.is_(None))
            .where(Artifact.stage == ArtifactStage.rendered)
            .where(Artifact.status == ArtifactStatus.done)
            .distinct(Label.model_uid)
            .order_by(Label.model_uid, Label.created_at.desc(), Label.id.desc())
        )
        rows = session.execute(stmt).all()

    # Drop anything outside the roster rather than trusting the write path. The
    # API validates class names now, but rows predate that check and the roster
    # itself can change; an unknown name would otherwise surface as a KeyError in
    # `CLASS_TO_INDEX` inside a DataLoader worker — mid-epoch, after a job has
    # already queued for a spot GPU and pulled a multi-GB image. Skipping is the
    # right call over raising: one stray row should not cancel a paid run, and
    # the count is printed so it cannot pass silently.
    samples = [(uid, class_name) for uid, class_name in rows if class_name in ROSTER]
    dropped = len(rows) - len(samples)
    if dropped:
        unknown_set = {class_name for _, class_name in rows if class_name not in ROSTER}
        print(f"skipped {dropped} label(s) outside the roster: {sorted(unknown_set)}")
    return samples


def data_snapshot(samples: list[tuple[str, str]], split: DatasetSplit) -> dict:
    """Capture *which data* this run trained on, for reproducibility (NFR-4).

    The `training_run.data_snapshot` blob: the count and per-class breakdown of the
    trainable set, a content `label_hash` over the sorted (uid, class) pairs so the
    set is identifiable (same hash → same data; a changed hash flags drift), the
    filter that produced it, and the train/val/test split sizes.
    """
    digest = hashlib.sha256()
    class_to_count: Counter[str] = Counter()
    for model_uid, class_name in sorted(samples):
        digest.update(f"{model_uid}\t{class_name}\n".encode())
        class_to_count[class_name] += 1

    return {
        "label_count": len(samples),
        "label_hash": "sha256:" + digest.hexdigest(),
        "as_of": datetime.now(UTC).isoformat(),
        "filter": {
            "deleted": "excluded",
            "label": "current (manual over weak)",
            "renders": "required",
        },
        "class_counts": dict(sorted(class_to_count.items())),
        "splits": split_sizes(split),
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
    run_id: int,
    step: int,
    loss: float,
    val_loss: float | None = None,
    val_accuracy: float | None = None,
) -> None:
    """Append one point to a run's loss curve (``training_metric``), committed on
    its own so the dashboard's cost curve grows while the run is still going.
    ``val_loss`` and ``val_accuracy`` are null on steps where validation was not
    evaluated."""
    with session_scope() as session:
        session.add(
            TrainingMetric(
                run_id=run_id,
                step=step,
                loss=loss,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
            )
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
    forever in ``running``."""
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


def _select_device(preference: str) -> torch.device:
    """Resolve the training device. ``"auto"`` prefers CUDA, then Apple MPS, then
    CPU; an explicit ``"cpu"``/``"cuda"``/``"mps"`` is honored as given."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_optimizer(config: Config, model: nn.Module) -> torch.optim.Optimizer:
    if config.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"unsupported optimizer {config.optimizer!r}; use 'adam' or 'sgd'")


def _build_loss(
    config: Config, train_samples: list[tuple[str, str]], device: torch.device
) -> nn.Module:
    """The training criterion, with optional class weighting and label smoothing.

    Weights are derived from the **training split only** — never the whole
    trainable set and never val/test, which would leak the evaluation splits'
    composition into training.

    The "balanced" formula is ``total / (num_classes * count)``, so a class of
    average size gets weight 1.0 and the weights stay centred around 1 rather
    than shrinking the overall loss scale (which would silently act as a
    learning-rate cut). A class absent from the training split gets weight 1.0:
    it is never a target, so its weight is unused, and 0.0 would be an equally
    unused value that merely looks alarming when the config is read back.
    """
    if config.class_weighting == "none":
        return nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    if config.class_weighting != "balanced":
        raise ValueError(
            f"unsupported class_weighting {config.class_weighting!r}; "
            "use 'none' or 'balanced'"
        )

    counts = Counter(class_name for _, class_name in train_samples)
    total = len(train_samples)
    weights = [
        total / (config.num_classes * counts[class_name]) if counts[class_name] else 1.0
        for class_name in ROSTER
    ]
    return nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        label_smoothing=config.label_smoothing,
    )


def _evaluate(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device
) -> tuple[float | None, float | None]:
    """Mean validation loss and accuracy over ``loader``, or ``(None, None)`` if
    the validation split is empty (which a very small local smoke can produce)."""
    if len(loader) == 0:
        return None, None
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for views, labels in loader:
            views = views.to(device)
            labels = labels.to(device)
            logits = model(views)
            total_loss += loss_fn(logits, labels).item() * labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += labels.size(0)
    return total_loss / total, correct / total


def _collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[list[int], list[int]]:
    """Every (true, predicted) class index over ``loader``, in loader order.

    Kept separate from ``_evaluate`` even though both do a forward pass: that one
    runs every epoch and only needs running totals, while this one materialises
    per-example labels for the end-of-run report. Folding them together would
    make the common path allocate two lists per epoch to serve a single use at
    the end.
    """
    model.eval()
    true_indices: list[int] = []
    predicted_indices: list[int] = []
    with torch.no_grad():
        for views, labels in loader:
            logits = model(views.to(device))
            predicted_indices.extend(logits.argmax(dim=1).cpu().tolist())
            true_indices.extend(labels.tolist())
    return true_indices, predicted_indices


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _save_weights(storage: Storage, key: str, model: nn.Module) -> None:
    """Write the model's state_dict to ``key`` (torch.save into an in-memory
    buffer, then one blob write through the storage abstraction)."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    storage.put_bytes(key, buffer.getvalue())


def run_training(
    config: Config, run_id: int, split: DatasetSplit
) -> tuple[str, dict | None]:
    """Train the multi-view CNN over the rendered views.

    Per-step training loss is logged (throttled by ``log_every``); once per epoch
    the validation split is evaluated for loss and accuracy, and the epoch's final
    step is logged with both, so the dashboard's cost curve shows the train/val
    gap and the accuracy series. The bookkeeping helpers (create_run / log_metric
    / finalize_run) are unchanged.

    Weights are checkpointed to the same key after every epoch (overwriting), so a
    spot preemption keeps the latest epoch.

    Returns ``(weights_key, report)``. The report is the B4 per-class evaluation
    (`ml/metrics.py`) computed once, after the final epoch, on the **validation**
    split — deliberately not the test split. Training already consults val every
    epoch, whereas test is held back for [evaluation](ml.md#evaluation) (M7);
    scoring test at the end of every run would erode it through repeated peeking
    long before M7 ever looked at it. ``None`` when the val split is empty, which
    a very small local smoke can produce.
    """
    torch.manual_seed(config.seed)  # seeded so shuffling/init are reproducible
    storage = build_storage(get_settings())
    device = _select_device(config.device)
    weights = weights_key(run_id)

    train_loader = DataLoader(
        MultiViewDataset(split.train, storage),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        MultiViewDataset(split.val, storage),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = MultiViewCNN.from_config(config).to(device)
    optimizer = _build_optimizer(config, model)
    loss_fn = _build_loss(config, split.train, device)

    global_step = 0
    for epoch in range(config.epochs):
        model.train()
        batch_count = len(train_loader)
        last_train_loss = 0.0
        for batch_index, (views, labels) in enumerate(train_loader):
            views = views.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(views)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            last_train_loss = loss.item()
            # Log on the interval, but skip the epoch's last step — it is logged
            # below together with the epoch's val loss, so each step is one row.
            is_last_batch = batch_index == batch_count - 1
            if not is_last_batch and global_step % config.log_every == 0:
                log_metric(run_id, global_step, last_train_loss)
            global_step += 1
        val_loss, val_accuracy = _evaluate(model, val_loader, loss_fn, device)
        # The epoch's final step carries its train loss and both val metrics.
        log_metric(run_id, global_step - 1, last_train_loss, val_loss, val_accuracy)
        _save_weights(storage, weights, model)  # checkpoint (overwrite) each epoch
        print(
            f"epoch {epoch + 1}/{config.epochs}  "
            f"train_loss={last_train_loss:.4f}  "
            f"val_loss={_format_metric(val_loss)}  "
            f"val_acc={_format_metric(val_accuracy)}"
        )

    # One pass at the end, on the trained model — not per epoch. The report is a
    # summary of the finished run, and computing it every epoch would add a full
    # extra forward pass over val for numbers nothing reads until the run ends.
    report = None
    if len(val_loader) > 0:
        true_indices, predicted_indices = _collect_predictions(model, val_loader, device)
        report = evaluation_report(true_indices, predicted_indices, ROSTER, split="val")
        print(
            f"val report: accuracy={_format_metric(report['accuracy'])}  "
            f"macro_recall={_format_metric(report['macro_recall'])}"
        )
    return weights, report


# --- Entry point -------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The CLI. Split from `_parse_args` so tests can inspect the accepted flags
    without running the trainer — the launch API builds these flag strings in a
    different file (server/app/api.py), and a mismatch there is only discovered
    ~15 min into a billed job.

    Run-time overrides for `Config`.

    The line is *what an experiment varies*, not every field on `Config`. Run
    shape (where it runs, how big, how much data), optimization and
    regularization are all here, because the launch page
    (web.md#starting-a-training-run) offers them and a web form cannot edit code.
    **Architecture stays code-edited** — backbone, view pooling and head shape
    change the checkpoint's shape, so a saved run's weights only load back into
    the architecture that produced them; keeping those fixed is what lets runs be
    compared and lets inference (M7) rebuild a model from a run's config.

    Every flag defaults to None so an unset one leaves the `Config` default alone
    rather than overwriting it with argparse's idea of a default.
    """
    parser = argparse.ArgumentParser(description="Train the multi-view CNN (M6).")
    parser.add_argument("--device", help='"cpu" | "cuda" | "mps" | "auto"')
    parser.add_argument("--num-workers", type=int, help="DataLoader worker processes")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--optimizer", help='"adam" | "sgd"')
    parser.add_argument("--momentum", type=float, help="SGD only")
    parser.add_argument("--dropout", type=float, help="dropout in the classifier head")
    parser.add_argument("--weight-decay", type=float, help="L2 penalty; 0 disables it")
    parser.add_argument(
        "--label-smoothing", type=float, help="softens the CE target; 0 disables it"
    )
    parser.add_argument(
        "--class-weighting",
        help=(
            '"none" | "balanced". "balanced" weights each class inversely to its '
            "frequency in the training split, so the 7.7:1 skew stops burying the "
            "tail classes."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help=(
            "train on a random subset of N models. The cost guardrail for a first "
            "cloud run: prove the wiring on a few hundred before paying for ~12k."
        ),
    )
    parser.add_argument("--notes", help="free-text description shown on the dashboard")
    return parser


def _parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _subsample(samples: list[tuple[str, str]], limit: int, seed: int) -> list[tuple[str, str]]:
    """A seeded random subset of `limit` samples.

    Proportional, not per-class balanced: it preserves the real class
    distribution (which is skewed ~7.7:1), so a small run rehearses the actual
    problem rather than an easier balanced version of it. Sorted first so the
    result depends on the sample *set* and the seed, never on query order.
    """
    if limit >= len(samples):
        return samples
    return random.Random(seed).sample(sorted(samples), limit)


def main() -> None:
    """Train one multi-view CNN run end to end: load the trainable set (labeled ∩
    rendered), split it, snapshot the data, open the run, train, and finalize. Any
    failure marks the run ``failed`` (so it never lingers as ``running``) and
    re-raises so the traceback is visible.

    Flags override the ``Config`` defaults; the loop and the bookkeeping are
    untouched by them, and whatever the flags resolve to is what gets recorded
    (NFR-4), so a cloud run stays as reproducible as a local one.
    """
    args = _parse_args()
    config = Config(
        **{
            name: value
            for name, value in vars(args).items()
            if name not in ("limit", "notes") and value is not None
        }
    )
    samples = load_trainable_samples()
    if not samples:
        raise SystemExit(
            "no trainable models: need models that are both labeled and rendered "
            "(run the pipeline and the weak-label backfill first)"
        )
    if args.limit is not None:
        samples = _subsample(samples, args.limit, config.seed)
    split = stratified_split(samples, config.seed)
    snapshot = data_snapshot(samples, split)
    if args.limit is not None:
        # The snapshot must say the run saw a *subset*, or its label_count reads
        # as the whole trainable set and two runs become falsely comparable.
        snapshot["limit"] = args.limit
    run_id = create_run(
        config,
        snapshot,
        notes=args.notes or f"{config.arch} baseline, {config.epochs} epochs",
    )
    sizes = snapshot["splits"]
    print(
        f"training_run {run_id}: {snapshot['label_count']} labeled+rendered models "
        f"(train {sizes['train']} / val {sizes['val']} / test {sizes['test']}), "
        f"{config.epochs} epochs on {config.device}"
    )
    try:
        weights_uri, report = run_training(config, run_id, split)
    except Exception:
        finalize_run(run_id, TrainingStatus.failed)
        raise
    finalize_run(
        run_id, TrainingStatus.completed, metrics=report, weights_uri=weights_uri
    )
    print(f"training_run {run_id}: completed, weights at {weights_uri}")


if __name__ == "__main__":
    main()
