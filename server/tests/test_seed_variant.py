"""The variant seeder — what it reads and what it refuses (backlog item 1).

One job here is one model through three stages and ~12 renders, so a defect is
paid for in Cloud Run time. The two things worth pinning are that a pilot is a
stable prefix of the full run (so the models eyeballed are the models that ran)
and that the default arm can never be republished by accident.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.artifact_keys import DEFAULT_VARIANT
from app.seed_variant import read_uids


def _write_csv(path: Path, uids: list[str]) -> Path:
    rows = "\n".join(f"{uid},chair" for uid in uids)
    path.write_text(f"uid,class\n{rows}\n", encoding="utf-8")
    return path


def test_a_pilot_is_a_prefix_of_the_full_run(tmp_path: Path) -> None:
    """The selections are written in sorted uid order, so the first N is stable:
    the eight models whose renders get eyeballed are the first eight of the real
    run, not eight others."""
    path = _write_csv(tmp_path / "subset.csv", ["a", "b", "c", "d"])

    assert read_uids([path], limit=2) == ["a", "b"]
    assert read_uids([path], limit=3)[:2] == read_uids([path], limit=2)


def test_both_selections_are_read_and_deduplicated(tmp_path: Path) -> None:
    """The training subset and the gold set are published together; a uid in both
    would publish twice and make the printed count a lie."""
    subset = _write_csv(tmp_path / "subset.csv", ["a", "b"])
    dev_set = _write_csv(tmp_path / "dev.csv", ["b", "c"])

    assert read_uids([subset, dev_set]) == ["a", "b", "c"]


def test_a_csv_without_a_uid_column_is_refused(tmp_path: Path) -> None:
    """Naming the wrong file publishes nothing rather than a KeyError per row."""
    path = tmp_path / "wrong.csv"
    path.write_text("model,class\na,chair\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="no `uid` column"):
        read_uids([path])


def test_a_missing_csv_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="nothing to publish"):
        read_uids([tmp_path / "absent.csv"])


def test_the_default_arm_is_not_a_choosable_variant() -> None:
    """Republishing `default` rewrites processed/converted/ and the artifact rows
    every catalog query reads — a reconciliation, not a seeding. The CLI refuses
    it; this pins the constant the refusal is written against."""
    assert DEFAULT_VARIANT == "default"
