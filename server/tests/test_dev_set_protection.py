"""Reserving dev-set models so they cannot be labeled (ml.md#the-second-dev-set).

Against real Postgres because the reservation is an upsert: the idempotency that
matters is `ON CONFLICT DO NOTHING`'s, and SQLite would not exercise it.

`build_dev_set` lives under `ml/`, which `pyproject.toml` puts on the path
alongside `server/`. The test lives here because what it is checking is a table
and its write path, not the selection logic.
"""

import pytest
from build_dev_set import mark_dev_set
from sqlalchemy import Engine, func, select, text

from app import db
from app.models import DevSetMember


@pytest.fixture
def clean_members(pg_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(db, "get_engine", lambda: pg_engine)
    with pg_engine.begin() as connection:
        connection.execute(text("TRUNCATE dev_set_member RESTART IDENTITY CASCADE"))


def _csv(tmp_path, rows: list[tuple[str, str]]):
    tmp_path.mkdir(parents=True, exist_ok=True)  # callers pass sub-dirs to get two CSVs
    path = tmp_path / "lvis_dev.csv"
    lines = ["uid,class,reason"]
    lines += [f"{uid},{class_name},lvis-gold" for uid, class_name in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _reserved(pg_engine: Engine) -> set[str]:
    with db.session_scope() as session:
        return set(session.scalars(select(DevSetMember.model_uid)).all())


def test_marks_every_uid_in_the_selection(clean_members, pg_engine, tmp_path) -> None:
    path = _csv(tmp_path, [("uid-a", "chair"), ("uid-b", "lamp"), ("uid-c", "table")])

    added = mark_dev_set(path)

    assert added == 3
    assert _reserved(pg_engine) == {"uid-a", "uid-b", "uid-c"}


def test_remarking_the_same_selection_adds_nothing(
    clean_members, pg_engine, tmp_path
) -> None:
    """NFR-2. Re-running a mark is the ordinary case — `make devset` marks on every
    selection — so a rerun must not duplicate or fail on the primary key."""
    path = _csv(tmp_path, [("uid-a", "chair"), ("uid-b", "lamp")])
    mark_dev_set(path)

    assert mark_dev_set(path) == 0
    assert _reserved(pg_engine) == {"uid-a", "uid-b"}


def test_a_grown_selection_adds_only_the_new_uids(
    clean_members, pg_engine, tmp_path
) -> None:
    mark_dev_set(_csv(tmp_path / "first", [("uid-a", "chair")]))

    added = mark_dev_set(_csv(tmp_path / "second", [("uid-a", "chair"), ("uid-b", "lamp")]))

    assert added == 1
    assert _reserved(pg_engine) == {"uid-a", "uid-b"}


def test_a_uid_dropped_from_the_csv_keeps_its_reservation(
    clean_members, pg_engine, tmp_path
) -> None:
    """Marking never *removes*. A uid leaving a rebuilt CSV does not mean nothing
    was ever scored against it, and un-reserving it would quietly make a model
    that has appeared in a published report labelable again."""
    mark_dev_set(_csv(tmp_path / "first", [("uid-a", "chair"), ("uid-b", "lamp")]))

    mark_dev_set(_csv(tmp_path / "second", [("uid-a", "chair")]))

    assert _reserved(pg_engine) == {"uid-a", "uid-b"}


def test_an_empty_selection_is_refused(clean_members, pg_engine, tmp_path) -> None:
    """Marking nothing would report success while protecting nothing — the same
    shape as the dev-set push that wrote to the local disk and said "pushed"."""
    with pytest.raises(SystemExit, match="no rows"):
        mark_dev_set(_csv(tmp_path, []))


def test_rows_outside_the_roster_are_not_reserved(
    clean_members, pg_engine, tmp_path
) -> None:
    """`load_dev_set` drops them, and a model that cannot be scored has no claim
    to protection."""
    path = _csv(tmp_path, [("uid-a", "chair"), ("uid-junk", "sofa")])

    mark_dev_set(path)

    assert _reserved(pg_engine) == {"uid-a"}


def test_two_dev_sets_reserve_independently(clean_members, pg_engine, tmp_path) -> None:
    path = _csv(tmp_path, [("uid-a", "chair")])
    mark_dev_set(path)

    added = mark_dev_set(path, name="second")

    assert added == 1
    with db.session_scope() as session:
        assert session.scalar(select(func.count()).select_from(DevSetMember)) == 2
