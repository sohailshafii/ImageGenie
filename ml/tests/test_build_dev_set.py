"""Second-dev-set selection (FR-7, ml.md#the-second-dev-set)."""

from types import SimpleNamespace

import build_dev_set
import pytest
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


# ── Reading and pushing the selection ────────────────────────────────────────
# The file is gitignored and a Vertex job has no checkout, so where it is read
# from decides whether `lvis` can be scored anywhere but a laptop.


class _FakeStorage:
    """Just enough Storage to see which key was written or read."""

    def __init__(self, contents: dict[str, bytes] | None = None) -> None:
        self.contents = contents or {}

    def put_bytes(self, key: str, data: bytes) -> None:
        self.contents[key] = data

    def get_bytes(self, key: str) -> bytes:
        return self.contents[key]  # KeyError where a real backend 404s


_CSV = "uid,class,reason\na,chair,lvis-gold\nb,lamp,lvis-gold\n"


def _gcs_settings():
    """Settings as they are wherever a push is legitimate — see push_dev_set."""
    return SimpleNamespace(storage_backend="gcs")


def test_the_local_file_wins_when_it_exists(tmp_path, monkeypatch) -> None:
    """Where a checkout has the selection, that is the copy being edited and
    regenerated — reading the bucket instead would silently score something else."""
    path = tmp_path / "lvis_dev.csv"
    path.write_text(_CSV, encoding="utf-8")
    stored = _FakeStorage({build_dev_set.dev_set_key("lvis"): b"uid,class\nz,car\n"})
    monkeypatch.setattr(build_dev_set, "build_storage", lambda _settings: stored)
    monkeypatch.setattr(build_dev_set, "get_settings", _gcs_settings)

    assert build_dev_set.load_dev_set(path) == [("a", "chair"), ("b", "lamp")]


def test_falls_back_to_the_stored_copy(tmp_path, monkeypatch) -> None:
    """The case the push exists for: a job with no `data/` directory."""
    stored = _FakeStorage({build_dev_set.dev_set_key("lvis"): _CSV.encode("utf-8")})
    monkeypatch.setattr(build_dev_set, "build_storage", lambda _settings: stored)
    monkeypatch.setattr(build_dev_set, "get_settings", _gcs_settings)

    samples = build_dev_set.load_dev_set(tmp_path / "absent.csv")

    assert samples == [("a", "chair"), ("b", "lamp")]


def test_missing_everywhere_names_both_places(tmp_path, monkeypatch) -> None:
    """Two different fixes — build the selection, or push it — so the message has
    to say which one is missing rather than just failing."""
    monkeypatch.setattr(build_dev_set, "build_storage", lambda _settings: _FakeStorage())
    monkeypatch.setattr(build_dev_set, "get_settings", _gcs_settings)

    with pytest.raises(SystemExit) as failure:
        build_dev_set.load_dev_set(tmp_path / "absent.csv")

    message = str(failure.value)
    assert "absent.csv" in message
    assert build_dev_set.dev_set_key("lvis") in message
    assert "devset-push" in message


def test_push_puts_the_file_where_a_job_reads_it(tmp_path, monkeypatch) -> None:
    path = tmp_path / "lvis_dev.csv"
    path.write_text(_CSV, encoding="utf-8")
    stored = _FakeStorage()
    monkeypatch.setattr(build_dev_set, "build_storage", lambda _settings: stored)
    monkeypatch.setattr(build_dev_set, "get_settings", _gcs_settings)

    key = build_dev_set.push_dev_set(path)

    assert key == build_dev_set.dev_set_key("lvis")
    # Byte-for-byte, and readable back by the loader — the two halves of the round
    # trip that has to hold for a cloud evaluation to score the same objects.
    assert stored.contents[key] == _CSV.encode("utf-8")
    assert build_dev_set.load_dev_set(tmp_path / "absent.csv") == [
        ("a", "chair"),
        ("b", "lamp"),
    ]


def test_pushing_a_selection_that_does_not_exist_is_refused(tmp_path) -> None:
    with pytest.raises(SystemExit, match="nothing to push"):
        build_dev_set.push_dev_set(tmp_path / "absent.csv")


def test_pushing_to_local_storage_is_refused(tmp_path, monkeypatch) -> None:
    """The backend defaults to `local`, where a "push" copies the file into
    data/storage/ and reports success while changing nothing a cloud job can
    read — a failure that would otherwise surface ~12 minutes into a paid job."""
    path = tmp_path / "lvis_dev.csv"
    path.write_text(_CSV, encoding="utf-8")
    monkeypatch.setattr(
        build_dev_set, "get_settings", lambda: SimpleNamespace(storage_backend="local")
    )

    with pytest.raises(SystemExit, match="no cloud job can read it"):
        build_dev_set.push_dev_set(path)
