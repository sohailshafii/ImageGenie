"""Use a finished training run: rebuild its model and classify with it.

The shared spine of two features that would otherwise duplicate each other —
the M7 evaluation over a held-out split (ml/evaluate.py) and the per-model
"what does the classifier think this is?" button (web.md). Both need the same
three steps: read a run's bookkeeping, rebuild the exact architecture it trained,
and load its weights.

**A run's config is the source of truth, never `Config`'s current defaults.**
Backbone, view pooling and head shape decide the *shape* of the saved tensors,
so weights only load back into the architecture that produced them. Reading
today's defaults instead would silently classify with a different model than the
one whose metrics you are looking at — or, more mercifully, fail on a shape
mismatch. This is what makes those fields deliberately absent from the launch
form (server.md#api-layer).
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from dataclasses import asdict
from types import SimpleNamespace

import torch
from dataset import MultiViewDataset, decode_view, load_views
from metrics import evaluation_report
from model import MultiViewCNN
from taxonomy import ROSTER
from torch.utils.data import DataLoader
from train import Config

from app.db import session_scope
from app.models import TrainingRun
from app.storage import Storage


class RunNotUsable(Exception):
    """A run cannot be loaded for inference — missing, or it never got as far as
    saving weights. Separate from a storage or shape error, which are bugs rather
    than states a caller should expect."""


def rebuild_config(stored: dict) -> SimpleNamespace:
    """A run's config as an attribute bag, tolerant of schema drift in both
    directions.

    Runs outlive the code that made them. An old run predates fields added since
    (run 2 has no `class_weighting`), and a run from a future branch may carry
    fields this checkout has dropped. So today's defaults fill what is missing and
    stored values always win, including keys `Config` no longer defines — they
    cost nothing here and keep the record faithful.

    Filling a missing *architecture* key is a guess, and deliberately an
    unguarded one: if the guess is wrong, `load_state_dict` fails on a shape
    mismatch, which is a far better outcome than quietly classifying with a
    silently different model.
    """
    return SimpleNamespace(**{**asdict(Config()), **stored})


def load_run_model(
    run_id: int, storage: Storage
) -> tuple[MultiViewCNN, SimpleNamespace, dict]:
    """Rebuild the model a run trained, with its weights loaded, in eval mode.

    Returns the model, its resolved config, and its data snapshot — the snapshot
    because a caller scoring a split needs the `label_hash` to check the data has
    not moved underneath it (ml/evaluate.py).
    """
    with session_scope() as session:
        run = session.get(TrainingRun, run_id)
        if run is None:
            raise RunNotUsable(f"no training run {run_id}")
        if not run.weights_uri:
            raise RunNotUsable(
                f"training run {run_id} has no saved weights (status {run.status.value})"
            )
        stored_config = dict(run.config or {})
        snapshot = dict(run.data_snapshot or {})
        weights_uri = run.weights_uri

    config = rebuild_config(stored_config)
    # Build the architecture *without* ImageNet initialisation, whatever the run
    # used. `pretrained` only chooses the backbone's starting weights, and the
    # checkpoint about to be loaded supplies every one of them — so honouring it
    # here downloads ~45 MB to overwrite it a line later, and makes inference
    # depend on reaching download.pytorch.org. The returned `config` keeps the
    # run's real value; only the object used to build the model differs.
    architecture = SimpleNamespace(**{**vars(config), "pretrained": False})
    model = MultiViewCNN.from_config(architecture)
    # map_location="cpu" because the weights were saved from a CUDA run and this
    # may well be a CPU box; weights_only because the blob is data, not code.
    state = torch.load(
        io.BytesIO(storage.get_bytes(weights_uri)), map_location="cpu", weights_only=True
    )
    model.load_state_dict(state)
    model.eval()
    return model, config, snapshot


@torch.no_grad()
def predict_probabilities(model: MultiViewCNN, views: torch.Tensor) -> torch.Tensor:
    """Class probabilities for a batch of stacked views, ``[batch, num_classes]``.

    Softmax rather than raw logits: every caller wants a confidence to show or
    threshold, and the ranking is identical either way.
    """
    return torch.softmax(model(views), dim=1)


def classify_model(
    model: MultiViewCNN, uid: str, storage: Storage
) -> list[tuple[str, float]]:
    """Classify one already-rendered model: (class_name, probability), best first.

    The whole roster is returned, not just the top class. A single label hides
    exactly the case worth seeing — a near-tie between `figure` and `animal`,
    where the model is not so much right as lucky.
    """
    return classify_views(model, load_views(storage, uid))


def views_from_images(images: list[bytes]) -> torch.Tensor:
    """Stack rendered PNGs into the ``[num_views, 3, H, W]`` tensor classification
    expects, applying the same decode and normalisation training used.

    The entry point for views that were never stored — rendered in memory from an
    upload — so the caller needs no knowledge of how a view becomes a tensor.
    """
    return torch.stack([decode_view(image) for image in images])


def rank_classes(probabilities: torch.Tensor) -> list[tuple[str, float]]:
    """One model's probability vector as (class_name, probability), best first."""
    return sorted(
        zip(ROSTER, probabilities.tolist(), strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )


def classify_views(
    model: MultiViewCNN, views: torch.Tensor
) -> list[tuple[str, float]]:
    """Classify one model from its stacked views, ``[num_views, 3, H, W]``.

    Takes the tensor rather than a uid so a caller holding views that were never
    stored — rendered in memory from an upload — uses the same path as one
    reading them from the bucket.
    """
    return rank_classes(predict_probabilities(model, views.unsqueeze(0))[0])


def rank_samples(
    model: MultiViewCNN,
    samples: list[tuple[str, str]],
    storage: Storage,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Iterator[tuple[str, str, list[tuple[str, float]]]]:
    """Rank the whole roster for each of `samples`, yielding (uid, class_name, ranked).

    The batched counterpart to `classify_model`, and the one to reach for over a
    whole split. Classifying model-by-model means twelve *sequential* storage
    reads per model with nothing overlapping them — measured at over 2 s/model
    against GCS, which is the read latency and not the forward pass. A DataLoader
    fans those reads across workers and hands the model a batch at a time, the
    same way `evaluate_samples` does.

    Yields in `samples` order (the loader does not shuffle), so a caller can pair
    each ranking with the sample it came from and report progress as it goes
    rather than after the last batch.
    """
    loader = DataLoader(
        MultiViewDataset(samples, storage),
        batch_size=batch_size,
        shuffle=False,  # so the results line up with `samples` positionally
        num_workers=num_workers,
    )
    ranked_samples = (
        rank_classes(probabilities)
        for views, _ in loader
        for probabilities in predict_probabilities(model, views)
    )
    for (uid, class_name), ranked in zip(samples, ranked_samples, strict=True):
        yield uid, class_name, ranked


def evaluate_samples(
    model: MultiViewCNN,
    samples: list[tuple[str, str]],
    storage: Storage,
    split_name: str,
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict:
    """Score `samples` and return the same report shape a run stores for `val`.

    Reuses `evaluation_report` so a test-set report and an end-of-run val report
    are directly comparable — the train/dev gap is only readable if both sides
    were computed the same way.
    """
    loader = DataLoader(
        MultiViewDataset(samples, storage),
        batch_size=batch_size,
        shuffle=False,  # order is irrelevant to the metrics and reproducible without it
        num_workers=num_workers,
    )
    predicted_indices: list[int] = []
    true_indices: list[int] = []
    for views, labels in loader:
        predicted_indices.extend(predict_probabilities(model, views).argmax(dim=1).tolist())
        true_indices.extend(labels.tolist())

    return evaluation_report(true_indices, predicted_indices, ROSTER, split_name)
