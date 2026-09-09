"""Multi-view dataset for training (M6 B5, ml.md#training).

Each sample is one model: its rendered views (from the processed bucket) stacked
into a ``[num_views, 3, H, W]`` tensor, plus its class index. Pixels are read
through the ``Storage`` abstraction — ``LocalStorage`` in dev, ``GcsStorage`` in
the cloud — so the same code trains locally on a handful of models and in the
cloud on the full set (NFR-5: bring the code to the data).
"""

from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image
from taxonomy import ROSTER
from torch.utils.data import Dataset

from app.artifact_keys import DEFAULT_VARIANT, view_keys
from app.storage import Storage

# The resnet backbone is ImageNet-pretrained, so inputs are normalized to the
# ImageNet channel statistics it was trained on.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# index i <-> ROSTER[i]; a run records the roster so this stays interpretable.
CLASS_TO_INDEX: dict[str, int] = {name: index for index, name in enumerate(ROSTER)}


def decode_view(data: bytes) -> torch.Tensor:
    """One view PNG as a normalized ``[3, H, W]`` float tensor.

    Split from the storage read so prediction can decode views it just rendered
    in memory, which never reach object storage (server/app/predict.py).
    """
    image = Image.open(io.BytesIO(data)).convert("RGB")
    pixels = np.asarray(image, dtype=np.float32) / 255.0  # H x W x 3, in [0, 1]
    tensor = torch.from_numpy(pixels).permute(2, 0, 1)  # -> 3 x H x W
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


def _load_view(storage: Storage, key: str) -> torch.Tensor:
    """Read one view PNG from storage and decode it."""
    return decode_view(storage.get_bytes(key))


def load_views(
    storage: Storage, uid: str, variant: str = DEFAULT_VARIANT
) -> torch.Tensor:
    """Every rendered view of one model, stacked into ``[num_views, 3, H, W]``.

    Standalone rather than inlined into `MultiViewDataset`, because inference
    (ml/infer.py) needs views for a model whose class is the unknown.

    ``variant`` chooses *which* renders — the shape-only default, or a parallel
    namespace such as `textured` (server.md#object-storage). It is the whole
    mechanism by which the two arms of the texture A/B differ: same models, same
    split, same config, different pixels. `layout_for` raises on an unknown
    variant rather than falling back, so a typo cannot quietly score one arm
    against the other's renders.
    """
    return torch.stack(
        [_load_view(storage, key) for key in view_keys(uid, variant)]
    )


class MultiViewDataset(Dataset):
    """One item per model: its stacked views and class index.

    ``samples`` is the ``(uid, class_name)`` list the caller has already resolved
    and split (ml/splits.py). This dataset only materializes pixels, so every
    model in ``samples`` must have all its views present. Filter first — the
    training script does that by querying the rendered ``artifact`` rows, far
    cheaper at full scale than a per-blob check; ``has_all_views`` below is the
    storage-level check used in the local smoke.
    """

    def __init__(
        self,
        samples: list[tuple[str, str]],
        storage: Storage,
        variant: str = DEFAULT_VARIANT,
    ) -> None:
        self._samples = samples
        self._storage = storage
        # Which render namespace the pixels come from. Carried on the dataset
        # rather than passed per item so one run can only ever read one arm.
        self._variant = variant

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        uid, class_name = self._samples[index]
        return (
            load_views(self._storage, uid, self._variant),
            CLASS_TO_INDEX[class_name],
        )


def has_all_views(
    storage: Storage, uid: str, variant: str = DEFAULT_VARIANT
) -> bool:
    """True if every rendered view for ``uid`` is present in storage.

    Used to skip half-rendered models in the local smoke so training never faults
    mid-epoch. At full scale, prefer filtering by the rendered ``artifact`` rows
    (one DB query) over this, which costs a storage HEAD per view.

    The variant arm has no ``artifact`` rows *by design* — writing them would
    overwrite the rows the control arm's trainability query depends on
    (server.md#object-storage) — so for a non-default variant this blob check is
    not the cheap fallback but the only answer available.
    """
    return all(storage.exists(key) for key in view_keys(uid, variant))
