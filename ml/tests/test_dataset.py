"""MultiViewDataset — render loading, normalization, label indexing (M6 B5)."""

import io
from pathlib import Path

import torch
from dataset import CLASS_TO_INDEX, MultiViewDataset, has_all_views
from PIL import Image
from taxonomy import ROSTER

from app.artifact_keys import NUM_VIEWS, view_key, view_keys
from app.storage import LocalStorage


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (224, 224), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_all_views(storage: LocalStorage, uid: str, color=(120, 120, 120)) -> None:
    for key in view_keys(uid):
        storage.put_bytes(key, _png_bytes(color))


def test_roster_indices_are_stable_and_cover_the_12_classes() -> None:
    assert len(ROSTER) == 12
    assert CLASS_TO_INDEX[ROSTER[0]] == 0
    assert sorted(CLASS_TO_INDEX.values()) == list(range(12))


def test_item_stacks_all_views_and_maps_the_label(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    _write_all_views(storage, "model-a")
    dataset = MultiViewDataset([("model-a", "chair")], storage)

    assert len(dataset) == 1
    views, label = dataset[0]
    assert views.shape == (NUM_VIEWS, 3, 224, 224)
    assert views.dtype == torch.float32
    assert label == CLASS_TO_INDEX["chair"]


def test_views_are_imagenet_normalized(tmp_path: Path) -> None:
    """A black frame (all zeros) normalizes to -mean/std on every channel, so
    every pixel is negative — proof the ImageNet normalization was applied."""
    storage = LocalStorage(tmp_path)
    _write_all_views(storage, "model-a", color=(0, 0, 0))
    views, _ = MultiViewDataset([("model-a", "lamp")], storage)[0]
    assert torch.all(views < 0)


def test_has_all_views_detects_a_missing_view(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    _write_all_views(storage, "complete")
    assert has_all_views(storage, "complete")

    # A model with only its first view rendered is not trainable.
    storage.put_bytes(view_key("partial", 0), _png_bytes((10, 10, 10)))
    assert not has_all_views(storage, "partial")
