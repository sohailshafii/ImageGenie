"""data_snapshot bookkeeping (M6 B5). The DB-backed loop is covered by the smoke."""

from splits import stratified_split
from train import data_snapshot


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
