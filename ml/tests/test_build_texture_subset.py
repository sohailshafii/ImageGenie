"""The texture A/B's subset selection (backlog item 1, ml.md#the-texture-ab).

Both arms of the experiment train on the list this module produces, so a defect
here does not fail — it moves the population under one arm and reports a number
that looks like a result. The property under test is therefore the one this
project has twice been burned by: **membership must be a pure function of the
model's own uid**, so that growing the candidate pool cannot reshuffle a
selection that two runs are already being compared on.
"""

from __future__ import annotations

from types import SimpleNamespace

import build_texture_subset
import pytest
from build_texture_subset import (
    load_qualifying_candidates,
    load_subset,
    push_subset,
    select_subset,
    subset_rank,
)
from taxonomy import ROSTER
from texture_census import CENSUS_HEADER, POPULATION_LVIS, POPULATION_TRAINABLE


def _census_row(uid: str, class_name: str, tier: str, population: str) -> str:
    """One census CSV line, positioned by the real header so the columns cannot drift."""
    values = {"uid": uid, "class": class_name, "tier": tier, "population": population}
    return ",".join(values.get(column, "0") for column in CENSUS_HEADER)


def _write_census(path, rows: list[tuple[str, str, str, str]]):
    lines = [",".join(CENSUS_HEADER)] + [_census_row(*row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_only_textured_trainable_models_in_the_roster_qualify(tmp_path) -> None:
    """The three filters, each of which silently widens the arm if it slips: a
    flat-coloured model would dilute the treatment with no surface detail to
    learn, an LVIS row is a *scoring* model the arms must not train on, and an
    off-roster class has no index to train against at all."""
    census = _write_census(
        tmp_path / "census.csv",
        [
            ("textured", "chair", "texture", POPULATION_TRAINABLE),
            ("flat", "chair", "uniform_colour", POPULATION_TRAINABLE),
            ("shapeless", "chair", "none", POPULATION_TRAINABLE),
            ("devset", "chair", "texture", POPULATION_LVIS),
            ("unlabeled", "unlabeled", "texture", POPULATION_TRAINABLE),
        ],
    )

    class_to_candidates = load_qualifying_candidates(census)

    assert class_to_candidates["chair"] == ["textured"]
    assert set(class_to_candidates) == set(ROSTER)
    assert sum(len(uids) for uids in class_to_candidates.values()) == 1


def test_a_missing_census_names_how_to_build_one(tmp_path) -> None:
    """`data/` is gitignored (NFR-6), so a fresh checkout has no census and the
    error has to say what produces one rather than a bare FileNotFoundError."""
    with pytest.raises(SystemExit, match="make census"):
        load_qualifying_candidates(tmp_path / "absent.csv")


def test_each_class_is_capped_and_a_short_class_gives_what_it_has() -> None:
    """Class-balanced with no redistribution: topping the total back up from the
    large classes is exactly the imbalance the cap exists to remove, and it would
    hide that `lamp` has ~10 test models however the draw is composed."""
    class_to_candidates = {class_name: [] for class_name in ROSTER}
    class_to_candidates["weapon"] = [f"weapon{index}" for index in range(10)]
    class_to_candidates["lamp"] = ["lamp0", "lamp1"]

    selected = select_subset(class_to_candidates, cap=4)

    assert sorted(class_name for _, class_name in selected) == ["lamp"] * 2 + ["weapon"] * 4
    assert selected == sorted(selected)


def test_adding_candidates_displaces_at_most_one_member() -> None:
    """The defect this selection is shaped to avoid. Index-based selection has
    twice destroyed a comparison here — one changed label moved 127 models across
    the test boundary, and a re-derived `--limit` subset scored 4 of run 17's 45
    held-out models at 0.0%. Hashing bounds the churn at the cap boundary, so a
    subset regenerated after more models are censused is still the same
    experiment."""
    original = [f"uid{index}" for index in range(20)]
    grown = original + ["newcomer"]
    cap = 5

    def ranked(uids: list[str]) -> dict[str, list[str]]:
        return {**{class_name: [] for class_name in ROSTER}, "chair": sorted(uids, key=subset_rank)}

    before = select_subset(ranked(original), cap)
    after = select_subset(ranked(grown), cap)

    lost = {uid for uid, _ in before} - {uid for uid, _ in after}
    assert len(lost) <= 1


def test_the_rank_is_salted_apart_from_the_split_and_the_subsample() -> None:
    """Two selection steps sharing a hash select in lockstep: a subset drawn on
    the split's hash would train on precisely what it is scored against, or on
    none of it. Cheap to assert, invisible if it regresses."""
    from splits import bucket_of
    from train import subsample_rank

    uid = "0002c6eafa154e8bb08ebafb715a8d46"
    assert subset_rank(uid) != subsample_rank(uid, 0)
    # Ordering, not just the digest: what matters is that the two steps rank the
    # same uids differently, which a shared salt would make identical.
    uids = [f"uid{index}" for index in range(200)]
    assert sorted(uids, key=subset_rank) != sorted(uids, key=lambda one: bucket_of(one, 0))


class _FakeStorage:
    """Just enough Storage to see which key was written, and to list it back."""

    def __init__(self) -> None:
        self.contents: dict[str, bytes] = {}

    def put_bytes(self, key: str, data: bytes) -> None:
        self.contents[key] = data

    def get_bytes(self, key: str) -> bytes:
        return self.contents[key]  # KeyError where a real backend 404s

    def list_sizes(self, prefix: str):
        return [
            (key, len(data)) for key, data in self.contents.items() if key.startswith(prefix)
        ]


_SUBSET_CSV = "uid,class\na,chair\nb,lamp\n"


def _gcs_settings():
    return SimpleNamespace(storage_backend="gcs")


def test_push_puts_the_subset_where_a_vertex_job_reads_it(tmp_path, monkeypatch, capsys) -> None:
    path = tmp_path / "textured_subset.csv"
    path.write_text(_SUBSET_CSV, encoding="utf-8")
    stored = _FakeStorage()
    monkeypatch.setattr(build_texture_subset, "build_storage", lambda _settings: stored)
    monkeypatch.setattr(build_texture_subset, "get_settings", _gcs_settings)

    key = push_subset(path, "textured_subset")

    assert key == build_texture_subset.experiment_subset_key("textured_subset")
    assert stored.contents[key] == _SUBSET_CSV.encode("utf-8")
    # The listing is the point: "pushed" has to be an observation of the bucket,
    # because a wrong bucket or a no-op push reads identically from the writer.
    assert key in capsys.readouterr().out.split("now holds")[1]


def test_pushing_a_subset_that_does_not_exist_is_refused(tmp_path) -> None:
    with pytest.raises(SystemExit, match="nothing to push"):
        push_subset(tmp_path / "absent.csv")


def test_pushing_to_local_storage_is_refused(tmp_path, monkeypatch) -> None:
    """`make devset-push` once printed "pushed" while writing to data/storage/ on
    the laptop. The same trap, and the same refusal."""
    path = tmp_path / "textured_subset.csv"
    path.write_text(_SUBSET_CSV, encoding="utf-8")
    monkeypatch.setattr(
        build_texture_subset,
        "get_settings",
        lambda: SimpleNamespace(storage_backend="local"),
    )

    with pytest.raises(SystemExit, match="no cloud job can read it"):
        push_subset(path)


def test_the_local_file_wins_when_it_exists(tmp_path, monkeypatch) -> None:
    """Where a checkout has the subset, that is the copy being regenerated and
    edited; reading the bucket instead would train on a stale population."""
    path = tmp_path / "textured_subset.csv"
    path.write_text(_SUBSET_CSV, encoding="utf-8")
    stored = _FakeStorage()
    stored.contents[build_texture_subset.experiment_subset_key("textured_subset")] = (
        b"uid,class\nz,car\n"
    )
    monkeypatch.setattr(build_texture_subset, "build_storage", lambda _settings: stored)
    monkeypatch.setattr(build_texture_subset, "get_settings", _gcs_settings)

    assert load_subset("textured_subset", path) == ["a", "b"]


def test_falls_back_to_the_stored_copy(tmp_path, monkeypatch) -> None:
    """The case the push exists for: a Vertex job with no `data/` directory."""
    stored = _FakeStorage()
    stored.contents[build_texture_subset.experiment_subset_key("textured_subset")] = (
        _SUBSET_CSV.encode("utf-8")
    )
    monkeypatch.setattr(build_texture_subset, "build_storage", lambda _settings: stored)
    monkeypatch.setattr(build_texture_subset, "get_settings", _gcs_settings)

    assert load_subset("textured_subset", tmp_path / "absent.csv") == ["a", "b"]


def test_missing_everywhere_names_both_places(tmp_path, monkeypatch) -> None:
    """Two different fixes — select the subset, or push it — so the error has to
    say which one is missing rather than just failing."""
    monkeypatch.setattr(
        build_texture_subset, "build_storage", lambda _settings: _FakeStorage()
    )
    monkeypatch.setattr(build_texture_subset, "get_settings", _gcs_settings)

    with pytest.raises(SystemExit) as failure:
        load_subset("textured_subset", tmp_path / "absent.csv")

    message = str(failure.value)
    assert "absent.csv" in message
    assert build_texture_subset.experiment_subset_key("textured_subset") in message
    assert "texture-subset-push" in message


def test_the_classes_are_not_loaded_back(tmp_path) -> None:
    """Uids only. A training run resolves each label through the live query, and
    returning the CSV's stale classes here would make training on a frozen copy of
    them an easy accident."""
    path = tmp_path / "textured_subset.csv"
    path.write_text(_SUBSET_CSV, encoding="utf-8")

    assert load_subset("textured_subset", path) == ["a", "b"]
