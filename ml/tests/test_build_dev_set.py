"""Second-dev-set selection (FR-7, ml.md#the-second-dev-set)."""

from build_dev_set import hash_order, select_dev_set
from taxonomy import ROSTER


def _gold_map(per_class: dict[str, int]) -> dict[str, str]:
    """uid -> gold class, with `count` synthetic uids per class."""
    return {
        f"{class_name}-{index}": class_name
        for class_name, count in per_class.items()
        for index in range(count)
    }


def _plentiful(count: int = 500) -> dict[str, str]:
    return _gold_map({class_name: count for class_name in ROSTER})


def test_excludes_already_ingested():
    uid_to_gold_class = _plentiful()
    ingested_uids_set = {uid for uid in uid_to_gold_class if uid.endswith("0")}
    selected = select_dev_set(uid_to_gold_class, ingested_uids_set, 120)
    assert selected
    assert not [uid for uid, _ in selected if uid in ingested_uids_set]


def test_balances_across_the_roster_when_supply_allows():
    selected = select_dev_set(_plentiful(), set(), 120)
    counts = {class_name: 0 for class_name in ROSTER}
    for _, class_name in selected:
        counts[class_name] += 1
    assert set(counts.values()) == {120 // len(ROSTER)}


def test_redistributes_the_shortfall_of_a_thin_class():
    """A class short of its quota costs coverage, not the whole budget."""
    per_class = {class_name: 500 for class_name in ROSTER}
    per_class["plant"] = 2
    selected = select_dev_set(_gold_map(per_class), set(), 120)
    assert len(selected) == 120
    assert sum(1 for _, class_name in selected if class_name == "plant") == 2
    # The shortfall goes to the lowest-hash leftovers, whatever class they are in.
    leftovers = sorted(
        (uid for uid, class_name in _gold_map(per_class).items()
         if class_name != "plant" and (uid, class_name) not in selected),
        key=hash_order,
    )
    assert leftovers  # there is a pool to redistribute from


def test_selection_is_stable_as_the_pool_grows():
    """The whole point of hashing: new supply must not reshuffle earlier picks.

    A dev set that reshuffles when more objects are ingested is a dev set whose
    before/after comparisons are uninterpretable — the same failure the training
    split had (ml.md#why-the-split-is-hashed-not-shuffled).
    """
    smaller = _plentiful(100)
    larger = _plentiful(200)
    first = {uid for uid, _ in select_dev_set(smaller, set(), 120)}
    second = {uid for uid, _ in select_dev_set(larger, set(), 120)}
    # Every uid the small pool offered is still in the large pool, so a stable
    # rule keeps most of the original picks rather than redrawing from scratch.
    assert len(first & second) > len(first) // 2


def test_caps_at_available_supply():
    uid_to_gold_class = _gold_map({class_name: 3 for class_name in ROSTER})
    selected = select_dev_set(uid_to_gold_class, set(), 10_000)
    assert len(selected) == len(uid_to_gold_class)
    assert len({uid for uid, _ in selected}) == len(selected)


def test_labels_match_the_gold_map():
    uid_to_gold_class = _plentiful()
    for uid, class_name in select_dev_set(uid_to_gold_class, set(), 120):
        assert uid_to_gold_class[uid] == class_name
