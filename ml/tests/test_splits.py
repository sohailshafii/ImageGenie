"""Deterministic stratified splits (M6 B5, ml.md#dataset-splits)."""

from splits import DatasetSplit, split_sizes, stratified_split


def _make_samples(per_class: dict[str, int]) -> list[tuple[str, str]]:
    return [
        (f"{class_name}-{index}", class_name)
        for class_name, count in per_class.items()
        for index in range(count)
    ]


def _all(split: DatasetSplit) -> list[tuple[str, str]]:
    return split.train + split.val + split.test


def test_same_seed_is_identical_different_seed_differs() -> None:
    samples = _make_samples({"chair": 100, "lamp": 80})
    assert stratified_split(samples, seed=0) == stratified_split(samples, seed=0)
    # A different seed shuffles differently, so the partition assignment changes.
    assert stratified_split(samples, seed=0) != stratified_split(samples, seed=1)


def test_input_order_does_not_change_the_split() -> None:
    samples = _make_samples({"chair": 50, "lamp": 50})
    reversed_samples = list(reversed(samples))
    assert stratified_split(samples, seed=7) == stratified_split(reversed_samples, seed=7)


def test_partitions_are_disjoint_and_cover_every_sample() -> None:
    samples = _make_samples({"chair": 100, "lamp": 80, "car": 60})
    split = stratified_split(samples, seed=3)
    combined = _all(split)
    assert len(combined) == len(samples)
    assert set(combined) == set(samples)  # nothing dropped or duplicated


def test_proportions_are_roughly_80_10_10_per_class() -> None:
    split = stratified_split(_make_samples({"chair": 100}), seed=0)
    assert split_sizes(split) == {"train": 80, "val": 10, "test": 10}


def test_every_class_appears_in_train() -> None:
    samples = _make_samples({"chair": 100, "lamp": 80, "car": 60})
    split = stratified_split(samples, seed=0)
    train_classes = {class_name for _, class_name in split.train}
    assert train_classes == {"chair", "lamp", "car"}


def test_a_tiny_class_falls_back_to_train_without_crashing() -> None:
    # 3 members: floor(0.1*3)=0 for both val and test, so all go to train.
    split = stratified_split(_make_samples({"plant": 3}), seed=0)
    assert split_sizes(split) == {"train": 3, "val": 0, "test": 0}
