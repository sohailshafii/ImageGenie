"""Selecting raw meshes to delete (server.md#reclaiming-raw-storage).

This is a destructive tool, so the tests that matter are the ones asserting what
it *keeps*. A model in the dataset must never appear as a candidate — losing its
mesh means re-downloading from Objaverse to get it back.
"""

from datetime import UTC, datetime

import pytest
from sqlalchemy import Engine, text

from app import db
from app.artifact_keys import raw_key
from app.cleanup_raw import (
    REASON_DELETED,
    REASON_ORPHAN,
    REASON_OUT_OF_SCOPE,
    REASON_UNLABELED,
    Candidate,
    delete_candidates,
    human_bytes,
    select_candidates,
    summarize,
)
from app.models import DownloadStatus, Label, LabelSource, Model

IN_SCOPE = ("chair", "lamp")


class FakeStorage:
    """In-memory Storage; `list_sizes` is the only method under test."""

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}
        self.deleted: list[str] = []

    def put_bytes(self, key: str, data: bytes) -> None:
        self.blobs[key] = data

    def exists(self, key: str) -> bool:
        return key in self.blobs

    def get_bytes(self, key: str) -> bytes:
        return self.blobs[key]

    def list_keys(self, prefix: str):
        return (key for key in sorted(self.blobs) if key.startswith(prefix))

    def list_sizes(self, prefix: str):
        return ((key, len(self.blobs[key])) for key in self.list_keys(prefix))

    def delete(self, key: str) -> None:
        self.deleted.append(key)
        self.blobs.pop(key, None)


@pytest.fixture
def storage() -> FakeStorage:
    return FakeStorage()


@pytest.fixture(autouse=True)
def clean_db(pg_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(db, "get_engine", lambda: pg_engine)
    with pg_engine.begin() as connection:
        connection.execute(text("TRUNCATE label, artifact, model RESTART IDENTITY CASCADE"))


def _add_model(uid: str, *, class_name: str | None = None, deleted: bool = False) -> None:
    with db.session_scope() as session:
        session.add(
            Model(
                uid=uid,
                download_status=DownloadStatus.downloaded,
                raw_key=raw_key(uid),
                deleted_at=datetime.now(UTC) if deleted else None,
            )
        )
        session.flush()  # no relationship(), so the FK insert must be ordered by hand
        if class_name is not None:
            session.add(
                Label(model_uid=uid, class_name=class_name, source=LabelSource.weak)
            )


def _candidates(storage: FakeStorage):
    with db.session_scope() as session:
        return list(select_candidates(session, storage, in_scope=IN_SCOPE))


def test_a_labeled_model_in_scope_is_kept(storage: FakeStorage) -> None:
    """The case that must never regress: losing this mesh costs a re-download."""
    _add_model("keeper", class_name="chair")
    storage.put_bytes(raw_key("keeper"), b"mesh")

    assert _candidates(storage) == []


def test_an_unlabeled_model_is_a_candidate(storage: FakeStorage) -> None:
    _add_model("unlabeled")
    storage.put_bytes(raw_key("unlabeled"), b"mesh")

    [candidate] = _candidates(storage)
    assert candidate.reason == REASON_UNLABELED
    assert candidate.uid == "unlabeled"


def test_a_soft_deleted_model_is_a_candidate_even_when_labeled(
    storage: FakeStorage,
) -> None:
    """Soft delete is the admin saying it isn't part of the dataset; the label
    it carried doesn't override that."""
    _add_model("removed", class_name="chair", deleted=True)
    storage.put_bytes(raw_key("removed"), b"mesh")

    [candidate] = _candidates(storage)
    assert candidate.reason == REASON_DELETED


def test_a_label_outside_the_roster_is_a_candidate(storage: FakeStorage) -> None:
    """`LabelIn.class_name` is a free-form str, so an out-of-scope class can be
    written through the API and would never be trained on."""
    _add_model("sofa", class_name="sofa")
    storage.put_bytes(raw_key("sofa"), b"mesh")

    [candidate] = _candidates(storage)
    assert candidate.reason == REASON_OUT_OF_SCOPE


def test_a_blob_with_no_model_row_is_a_candidate(storage: FakeStorage) -> None:
    storage.put_bytes(raw_key("ghost"), b"mesh")

    [candidate] = _candidates(storage)
    assert candidate.reason == REASON_ORPHAN


def test_a_manual_label_overrides_an_earlier_weak_one(storage: FakeStorage) -> None:
    """The current label is manual-wins-over-weak, same as the API resolves it —
    so correcting an out-of-scope weak label back into the roster saves the mesh."""
    _add_model("corrected", class_name="sofa")
    with db.session_scope() as session:
        session.add(
            Label(model_uid="corrected", class_name="lamp", source=LabelSource.manual)
        )
    storage.put_bytes(raw_key("corrected"), b"mesh")

    assert _candidates(storage) == []


def test_an_unparseable_key_is_never_a_candidate(storage: FakeStorage) -> None:
    """A key we cannot map to a uid is one we cannot claim to own, so it is left
    alone rather than deleted — the reconciler treats strays the same way."""
    storage.put_bytes("raw/nested/unexpected.glb", b"mystery")
    storage.put_bytes("raw/README.txt", b"notes")

    assert _candidates(storage) == []


def test_processed_artifacts_are_never_considered(storage: FakeStorage) -> None:
    """The tool only ever lists raw/. Renders are what the UI and training read."""
    _add_model("unlabeled")
    storage.put_bytes(raw_key("unlabeled"), b"mesh")
    storage.put_bytes("processed/renders/unlabeled/view_00.png", b"png")
    storage.put_bytes("processed/normalized/unlabeled.ply", b"ply")

    assert [candidate.key for candidate in _candidates(storage)] == [raw_key("unlabeled")]


def test_summarize_groups_by_reason(storage: FakeStorage) -> None:
    _add_model("keeper", class_name="chair")
    _add_model("unlabeled-a")
    _add_model("unlabeled-b")
    _add_model("removed", class_name="lamp", deleted=True)
    for uid in ("keeper", "unlabeled-a", "unlabeled-b", "removed"):
        storage.put_bytes(raw_key(uid), b"0123456789")
    storage.put_bytes(raw_key("ghost"), b"01234")

    counts, total_bytes = summarize(_candidates(storage))

    assert counts[REASON_UNLABELED] == 2
    assert counts[REASON_DELETED] == 1
    assert counts[REASON_ORPHAN] == 1
    assert total_bytes[REASON_UNLABELED] == 20
    assert total_bytes[REASON_ORPHAN] == 5
    assert sum(counts.values()) == 4  # the keeper is not among them


# ── Deleting ────────────────────────────────────────────────────────────────


def test_delete_removes_only_the_candidates(storage: FakeStorage) -> None:
    _add_model("keeper", class_name="chair")
    _add_model("unlabeled")
    for uid in ("keeper", "unlabeled"):
        storage.put_bytes(raw_key(uid), b"mesh")
    storage.put_bytes("processed/renders/keeper/view_00.png", b"png")

    deleted_count, reclaimed = delete_candidates(storage, _candidates(storage))

    assert deleted_count == 1
    assert reclaimed == 4
    assert storage.deleted == [raw_key("unlabeled")]
    assert storage.exists(raw_key("keeper"))
    assert storage.exists("processed/renders/keeper/view_00.png")


def test_delete_is_rerunnable_after_an_interruption(storage: FakeStorage) -> None:
    """Storage.delete tolerates an absent key, so a half-finished run can just be
    repeated rather than needing to know where it stopped (NFR-2)."""
    _add_model("unlabeled")
    storage.put_bytes(raw_key("unlabeled"), b"mesh")
    candidates = _candidates(storage)

    delete_candidates(storage, candidates)
    deleted_count, reclaimed = delete_candidates(storage, candidates)  # same list again

    assert deleted_count == 1  # reported as deleted; the blob was already gone
    assert not storage.exists(raw_key("unlabeled"))


def test_one_failure_does_not_abort_the_run(
    storage: FakeStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finishing the rest and re-running beats stopping halfway with no record."""
    candidates = [
        Candidate(raw_key("first"), "first", REASON_ORPHAN, 10),
        Candidate(raw_key("boom"), "boom", REASON_ORPHAN, 10),
        Candidate(raw_key("third"), "third", REASON_ORPHAN, 10),
    ]

    def explode_on_boom(key: str) -> None:
        if "boom" in key:
            raise RuntimeError("storage is having a day")
        storage.deleted.append(key)

    monkeypatch.setattr(storage, "delete", explode_on_boom)

    deleted_count, reclaimed = delete_candidates(storage, candidates)

    assert deleted_count == 2
    assert reclaimed == 20
    assert storage.deleted == [raw_key("first"), raw_key("third")]


def test_nothing_to_delete_is_not_an_error(storage: FakeStorage) -> None:
    _add_model("keeper", class_name="chair")
    storage.put_bytes(raw_key("keeper"), b"mesh")

    assert delete_candidates(storage, _candidates(storage)) == (0, 0)


@pytest.mark.parametrize(
    ("size", "expected"),
    [(0, "0 B"), (512, "512 B"), (1536, "1.5 KB"), (5 * 1024**3, "5.0 GB")],
)
def test_human_bytes(size: int, expected: str) -> None:
    assert human_bytes(size) == expected
