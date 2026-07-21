"""Weak-label backfill (server.md#weak-label-backfill).

The load itself is simple; what these cover is the three ways it can go wrong —
duplicating rows on a rerun, tripping the model FK, or trampling a human's
correction.
"""

import json

import pytest
from sqlalchemy import Engine, select, text

from app import db
from app.backfill_labels import backfill
from app.models import DownloadStatus, Label, LabelSource, Model

EVAL_FIXTURE = {
    "per_class_metrics": {
        "chair": {"precision": 0.972, "recall": 0.389},
        "car": {"precision": 0.964, "recall": 0.641},
        "figure": {"precision": 0.622, "recall": 0.220},
    }
}


@pytest.fixture
def seeded(pg_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    """Three downloaded models — the CSV will reference a fourth that doesn't exist."""
    monkeypatch.setattr(db, "get_engine", lambda: pg_engine)
    with pg_engine.begin() as connection:
        connection.execute(
            text("TRUNCATE label, artifact, model, app_user RESTART IDENTITY CASCADE")
        )
    with db.session_scope() as session:
        for uid in ("uid-chair", "uid-car", "uid-figure"):
            session.add(Model(uid=uid, download_status=DownloadStatus.downloaded))


@pytest.fixture
def labels_csv(tmp_path):
    path = tmp_path / "weak_labels.csv"
    path.write_text(
        "uid,class,reason\n"
        "uid-chair,chair,category\n"
        "uid-car,car,keyword\n"
        "uid-figure,figure,rescue\n"
        "uid-never-downloaded,chair,keyword\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def eval_json(tmp_path):
    path = tmp_path / "weak_label_eval.json"
    path.write_text(json.dumps(EVAL_FIXTURE), encoding="utf-8")
    return path


def _labels_by_uid() -> dict[str, Label]:
    with db.session_scope() as session:
        rows = session.scalars(select(Label)).all()
        return {
            row.model_uid: (row.class_name, row.source, row.confidence, row.annotator)
            for row in rows
        }


def test_inserts_a_weak_label_per_downloaded_model(seeded, labels_csv, eval_json) -> None:
    counts = backfill(labels_csv, eval_json, dry_run=False)

    assert counts["inserted"] == 3
    # The fourth uid has no model row — expected, since the CSV covers the whole
    # labeled set while the DB only holds what finished downloading.
    assert counts["missing_model"] == 1

    labels = _labels_by_uid()
    assert labels["uid-chair"][0] == "chair"
    assert labels["uid-chair"][1] is LabelSource.weak
    assert labels["uid-chair"][3] is None  # rule-derived: no annotator


def test_confidence_is_the_measured_precision(seeded, labels_csv, eval_json) -> None:
    """Not an invented number — it's how often that class's weak label is right,
    so lowest-confidence-first is a meaningful review order."""
    backfill(labels_csv, eval_json, dry_run=False)
    labels = _labels_by_uid()
    assert labels["uid-figure"][2] == pytest.approx(0.622)  # worst class
    assert labels["uid-chair"][2] == pytest.approx(0.972)
    assert labels["uid-figure"][2] < labels["uid-car"][2]


def test_rerunning_inserts_nothing_new(seeded, labels_csv, eval_json) -> None:
    """NFR-2: reruns must skip already-processed rows."""
    backfill(labels_csv, eval_json, dry_run=False)
    second = backfill(labels_csv, eval_json, dry_run=False)

    assert second["inserted"] == 0
    assert second["skipped"] == 3
    with db.session_scope() as session:
        assert len(session.scalars(select(Label)).all()) == 3  # no duplicates


def test_a_manual_correction_survives_a_rerun(seeded, labels_csv, eval_json) -> None:
    """The whole point of the skip: re-importing must not undo human work."""
    backfill(labels_csv, eval_json, dry_run=False)
    with db.session_scope() as session:
        session.add(
            Label(
                model_uid="uid-figure",
                class_name="animal",  # a human fixing the figure/animal boundary
                source=LabelSource.manual,
                annotator="admin@imagegenie.dev",
            )
        )

    backfill(labels_csv, eval_json, dry_run=False)

    with db.session_scope() as session:
        figure_labels = session.scalars(
            select(Label).where(Label.model_uid == "uid-figure").order_by(Label.id)
        ).all()
        # Still exactly one weak + one manual, and the manual is last, so the
        # API's "most recent wins" keeps resolving it as current.
        assert [label.source for label in figure_labels] == [
            LabelSource.weak,
            LabelSource.manual,
        ]
        assert figure_labels[-1].class_name == "animal"


def test_duplicate_csv_rows_insert_once(seeded, tmp_path, eval_json) -> None:
    path = tmp_path / "dupes.csv"
    path.write_text(
        "uid,class,reason\nuid-chair,chair,category\nuid-chair,chair,keyword\n",
        encoding="utf-8",
    )
    counts = backfill(path, eval_json, dry_run=False)
    assert counts["inserted"] == 1
    with db.session_scope() as session:
        assert len(session.scalars(select(Label)).all()) == 1


def test_dry_run_writes_nothing(seeded, labels_csv, eval_json) -> None:
    counts = backfill(labels_csv, eval_json, dry_run=True)
    assert counts["inserted"] == 3  # what it *would* do
    with db.session_scope() as session:
        assert session.scalars(select(Label)).all() == []


def test_missing_eval_file_still_loads_with_null_confidence(
    seeded, labels_csv, tmp_path, caplog
) -> None:
    with caplog.at_level("WARNING", logger="app.backfill_labels"):
        counts = backfill(labels_csv, tmp_path / "absent.json", dry_run=False)
    assert counts["inserted"] == 3
    assert all(label[2] is None for label in _labels_by_uid().values())
    assert "NULL confidence" in caplog.text


def test_classes_missing_from_the_eval_are_reported(seeded, labels_csv, tmp_path, caplog) -> None:
    """A CSV class the eval never scored means the two artifacts are from
    different runs — that should be said out loud, not silently written as NULL."""
    partial = tmp_path / "partial.json"
    partial.write_text(
        json.dumps({"per_class_metrics": {"chair": {"precision": 0.9}}}), encoding="utf-8"
    )
    with caplog.at_level("WARNING", logger="app.backfill_labels"):
        backfill(labels_csv, partial, dry_run=False)
    assert "car" in caplog.text and "figure" in caplog.text
