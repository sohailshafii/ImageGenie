"""Deterministic hash-bucketed splits (M6 B5, ml.md#dataset-splits)."""

from splits import DatasetSplit, split_sizes, stratified_split


def _make_samples(per_class: dict[str, int]) -> list[tuple[str, str]]:
    return [
        (f"{class_name}-{index}", class_name)
        for class_name, count in per_class.items()
        for index in range(count)
    ]


def _all(split: DatasetSplit) -> list[tuple[str, str]]:
    return split.train + split.val + split.test


def _partition_of(split: DatasetSplit) -> dict[str, str]:
    """uid -> partition name, for comparing two splits model by model."""
    return {
        uid: name
        for name, members in (
            ("train", split.train), ("val", split.val), ("test", split.test)
        )
        for uid, _ in members
    }


def test_correcting_a_label_moves_nothing_between_partitions() -> None:
    """The property the whole module exists for.

    The previous shuffle-and-slice implementation made the partition a function of
    the entire labeled set, so one corrected label re-randomised most of the
    corpus — which left runs either side of a correction pass sharing only 29% of
    their test sets and unable to be compared (ml.md#evaluation).
    """
    samples = _make_samples(dict.fromkeys(("animal", "figure", "weapon", "chair"), 200))
    corrected = [
        (uid, "figure" if uid == "animal-0" else class_name) for uid, class_name in samples
    ]

    before = _partition_of(stratified_split(samples, seed=0))
    after = _partition_of(stratified_split(corrected, seed=0))

    assert before == after  # the relabeled model included: its class moved, not its bucket


def test_removing_models_leaves_the_survivors_where_they_were() -> None:
    """Soft-deleting or un-rendering models must not reshuffle the rest either —
    the corpus shrinks over time and evaluations still have to line up."""
    samples = _make_samples({"chair": 300, "lamp": 300})
    smaller = [pair for pair in samples if not pair[0].endswith(("1", "7"))]

    full = _partition_of(stratified_split(samples, seed=0))
    reduced = _partition_of(stratified_split(smaller, seed=0))

    assert all(full[uid] == partition for uid, partition in reduced.items())


def test_same_seed_is_identical_different_seed_differs() -> None:
    samples = _make_samples({"chair": 100, "lamp": 80})
    assert stratified_split(samples, seed=0) == stratified_split(samples, seed=0)
    # The seed is hashed with the uid, so it still selects a different partition.
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


def test_proportions_are_near_80_10_10() -> None:
    """Approximate, not exact — that is the trade hashing makes. At roster scale
    the drift is a handful of models, and every run records its actual sizes."""
    split = stratified_split(_make_samples({"chair": 4000}), seed=0)
    sizes = split_sizes(split)

    assert sizes["train"] + sizes["val"] + sizes["test"] == 4000
    assert 0.75 < sizes["train"] / 4000 < 0.85
    assert 0.07 < sizes["val"] / 4000 < 0.13
    assert 0.07 < sizes["test"] / 4000 < 0.13


def test_each_class_is_represented_across_partitions() -> None:
    """The hash is independent of class, so every class lands in every partition
    at the same expected rate — stratification without pinning exact counts."""
    samples = _make_samples({"chair": 400, "lamp": 400, "car": 400})
    split = stratified_split(samples, seed=0)

    for members in (split.train, split.val, split.test):
        assert {class_name for _, class_name in members} == {"chair", "lamp", "car"}


def test_a_tiny_class_does_not_crash() -> None:
    """Too small to fill a 10% slot, so its members land wherever they hash —
    fine for a local smoke, and it must not raise."""
    split = stratified_split(_make_samples({"plant": 3}), seed=0)

    assert sum(split_sizes(split).values()) == 3
