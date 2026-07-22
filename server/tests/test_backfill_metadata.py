"""Objaverse metadata backfill (server.md#metadata-backfill).

`objaverse` is stubbed throughout — the real call downloads hundreds of MB of
shard files, which no test should do.
"""

import pytest
from sqlalchemy import Engine, select, text

from app import backfill_metadata, db
from app.backfill_metadata import backfill, extract_metadata
from app.models import DownloadStatus, Model

ANNOTATIONS = {
    "uid-chair": {
        "name": "Vintage Wooden Chair",
        "tags": [{"name": "furniture"}, {"name": "wood"}],
        "categories": [{"name": "furniture-home"}],
    },
    "uid-car": {"name": "Red Sports Car", "tags": [{"name": "vehicle"}]},
    "uid-blank": {"name": "", "tags": [], "categories": []},
}


@pytest.fixture
def seeded(pg_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(db, "get_engine", lambda: pg_engine)
    with pg_engine.begin() as connection:
        connection.execute(
            text("TRUNCATE label, artifact, model, app_user RESTART IDENTITY CASCADE")
        )
    with db.session_scope() as session:
        for uid in ("uid-chair", "uid-car", "uid-blank", "uid-unknown"):
            session.add(Model(uid=uid, download_status=DownloadStatus.downloaded))


@pytest.fixture
def fake_objaverse(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Stub `objaverse.load_annotations`; records the uid batches it was asked for."""
    requested: list[list[str]] = []

    class FakeObjaverse:
        @staticmethod
        def load_annotations(uids):
            requested.append(list(uids))
            return {uid: ANNOTATIONS[uid] for uid in uids if uid in ANNOTATIONS}

    monkeypatch.setitem(__import__("sys").modules, "objaverse", FakeObjaverse)
    return requested


# ── Mapping ─────────────────────────────────────────────────────────────────
def test_extract_folds_categories_in_with_tags() -> None:
    title, tags = extract_metadata(ANNOTATIONS["uid-chair"])
    assert title == "Vintage Wooden Chair"
    assert tags == ["furniture", "wood", "furniture-home"]


def test_extract_dedupes_and_drops_blanks() -> None:
    _, tags = extract_metadata(
        {"tags": [{"name": "wood"}, {"name": ""}, {}, {"name": "wood"}], "categories": None}
    )
    assert tags == ["wood"]


def test_extract_treats_an_empty_name_as_no_title() -> None:
    title, _ = extract_metadata({"name": "   "})
    assert title is None


# ── Backfill ────────────────────────────────────────────────────────────────
def _titles() -> dict[str, tuple]:
    with db.session_scope() as session:
        return {
            model.uid: (model.title, model.tags)
            for model in session.scalars(select(Model)).all()
        }


def test_backfill_stores_title_and_tags(seeded, fake_objaverse) -> None:
    counts = backfill(limit=None, dry_run=False)

    assert counts["updated"] == 2  # chair + car
    stored = _titles()
    assert stored["uid-chair"] == ("Vintage Wooden Chair", ["furniture", "wood", "furniture-home"])
    assert stored["uid-car"][0] == "Red Sports Car"


def test_models_without_a_usable_annotation_are_counted_not_failed(
    seeded, fake_objaverse
) -> None:
    """One uid the shard doesn't know, one whose annotation is empty — neither
    should abort a 30k-row run."""
    counts = backfill(limit=None, dry_run=False)
    assert counts["no_annotation"] == 2
    assert _titles()["uid-unknown"] == (None, None)


def test_rerun_only_fetches_models_still_missing_metadata(seeded, fake_objaverse) -> None:
    """NFR-2: a rerun after a partial pass resumes rather than refetching."""
    backfill(limit=None, dry_run=False)
    fake_objaverse.clear()

    counts = backfill(limit=None, dry_run=False)

    assert counts["updated"] == 0
    # The two that succeeded are no longer requested; only the unusable remain.
    assert sorted(fake_objaverse[0]) == ["uid-blank", "uid-unknown"]


def test_limit_caps_the_batch(seeded, fake_objaverse) -> None:
    counts = backfill(limit=1, dry_run=False)
    assert counts["pending"] == 1


def test_dry_run_fetches_but_writes_nothing(seeded, fake_objaverse) -> None:
    counts = backfill(limit=None, dry_run=True)
    assert counts["updated"] == 2  # what it would have written
    assert all(title is None for title, _ in _titles().values())


def test_nothing_to_do_makes_no_fetch(seeded, fake_objaverse) -> None:
    """The expensive part is the shard download — don't trigger it for an empty
    work list."""
    with db.session_scope() as session:
        for model in session.scalars(select(Model)).all():
            model.title = "already set"

    counts = backfill(limit=None, dry_run=False)

    assert counts["pending"] == 0
    assert fake_objaverse == []


def test_batches_are_bounded(seeded, fake_objaverse, monkeypatch) -> None:
    monkeypatch.setattr(backfill_metadata, "FETCH_BATCH_SIZE", 2)
    backfill(limit=None, dry_run=False)
    assert [len(batch) for batch in fake_objaverse] == [2, 2]
