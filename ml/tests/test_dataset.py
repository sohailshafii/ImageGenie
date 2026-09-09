"""MultiViewDataset — render loading, normalization, label indexing (M6 B5)."""

import io
from pathlib import Path

import torch
from dataset import CLASS_TO_INDEX, MultiViewDataset, has_all_views
from PIL import Image
from taxonomy import ROSTER

from app.artifact_keys import (
    NUM_VIEWS,
    TEXTURED_VARIANT,
    view_key,
    view_keys,
)
from app.storage import LocalStorage


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (224, 224), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_all_views(
    storage: LocalStorage, uid: str, color=(120, 120, 120), variant: str | None = None
) -> None:
    keys = view_keys(uid) if variant is None else view_keys(uid, variant)
    for key in keys:
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


def test_a_variant_reads_its_own_renders(tmp_path: Path) -> None:
    """The A/B's entire mechanism: same model, same label, different pixels. The
    two arms exist as parallel key namespaces precisely so one cannot overwrite
    the other, and this is the read side of that — a dataset built on `textured`
    must not fall back to the shape-only renders when it is the treatment arm."""
    storage = LocalStorage(tmp_path)
    _write_all_views(storage, "model-a", color=(10, 10, 10))
    _write_all_views(storage, "model-a", color=(200, 40, 40), variant=TEXTURED_VARIANT)

    shape_only, _ = MultiViewDataset([("model-a", "chair")], storage)[0]
    textured, _ = MultiViewDataset(
        [("model-a", "chair")], storage, TEXTURED_VARIANT
    )[0]

    assert shape_only.shape == textured.shape
    assert not torch.allclose(shape_only, textured)


def test_a_variant_with_no_renders_is_not_silently_the_control_arm(tmp_path: Path) -> None:
    """The failure this has to make loud. A treatment-arm model whose textured
    renders were never written would, under any fallback, train on the control
    arm's grey pixels and report a number that looks like a result."""
    storage = LocalStorage(tmp_path)
    _write_all_views(storage, "model-a")

    assert has_all_views(storage, "model-a")
    assert not has_all_views(storage, "model-a", TEXTURED_VARIANT)
