"""data_snapshot bookkeeping (M6 B5). The DB-backed loop is covered by the smoke."""

import io
from pathlib import Path

import torch
from splits import stratified_split
from torch import nn
from train import _save_weights, data_snapshot

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
