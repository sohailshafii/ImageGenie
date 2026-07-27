"""data_snapshot bookkeeping (M6 B5). The DB-backed loop is covered by the smoke."""

import io
from pathlib import Path

import pytest
import torch
from splits import stratified_split
from taxonomy import ROSTER
from torch import nn
from train import Config, _build_loss, _save_weights, data_snapshot

from app.artifact_keys import weights_key
from app.storage import LocalStorage


def test_data_snapshot_records_counts_hash_filter_and_splits() -> None:
    samples = [("b", "chair"), ("a", "chair"), ("c", "lamp")]
    split = stratified_split(samples, seed=0)
    snapshot = data_snapshot(samples, split)

    assert snapshot["label_count"] == 3
    assert snapshot["class_counts"] == {"chair": 2, "lamp": 1}
    assert snapshot["label_hash"].startswith("sha256:")
    assert snapshot["filter"]["renders"] == "required"
    assert set(snapshot["splits"]) == {"train", "val", "test"}
    assert sum(snapshot["splits"].values()) == 3


def test_label_hash_is_order_independent() -> None:
    samples = [("a", "chair"), ("b", "lamp"), ("c", "car")]
    split = stratified_split(samples, seed=0)
    forward = data_snapshot(samples, split)["label_hash"]
    reverse = data_snapshot(list(reversed(samples)), split)["label_hash"]
    assert forward == reverse  # sorted before hashing, so input order can't matter


def test_weights_key_is_a_processed_bucket_path() -> None:
    assert weights_key(7) == "processed/models/7.pt"


def test_save_weights_round_trips_through_storage(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    model = nn.Linear(4, 3)
    key = weights_key(42)

    _save_weights(storage, key, model)

    assert storage.exists(key)
    reloaded = torch.load(io.BytesIO(storage.get_bytes(key)), weights_only=True)
    assert set(reloaded) == set(model.state_dict())


def test_unweighted_loss_is_plain_cross_entropy() -> None:
    loss_fn = _build_loss(Config(), [("a", "chair")], torch.device("cpu"))

    assert loss_fn.weight is None
    assert loss_fn.label_smoothing == 0.0


def test_balanced_weighting_is_inverse_to_training_frequency() -> None:
    # 6 weapons to 2 chairs in a 12-class roster: total/(num_classes*count)
    # puts chair 3x above weapon, and leaves absent classes at 1.0 (unused).
    samples = [(f"w{index}", "weapon") for index in range(6)]
    samples += [(f"c{index}", "chair") for index in range(2)]
    config = Config(class_weighting="balanced")

    weights = _build_loss(config, samples, torch.device("cpu")).weight

    weight_of = {name: weights[index].item() for index, name in enumerate(ROSTER)}
    assert weight_of["chair"] == pytest.approx(8 / (12 * 2))
    assert weight_of["weapon"] == pytest.approx(8 / (12 * 6))
    assert weight_of["chair"] == pytest.approx(3 * weight_of["weapon"])
    assert weight_of["aircraft"] == 1.0  # absent from training, never a target


def test_label_smoothing_reaches_the_criterion() -> None:
    loss_fn = _build_loss(
        Config(label_smoothing=0.1), [("a", "chair")], torch.device("cpu")
    )

    assert loss_fn.label_smoothing == 0.1


def test_unknown_class_weighting_is_rejected() -> None:
    config = Config(class_weighting="inverse-sqrt")

    with pytest.raises(ValueError, match="unsupported class_weighting"):
        _build_loss(config, [("a", "chair")], torch.device("cpu"))
