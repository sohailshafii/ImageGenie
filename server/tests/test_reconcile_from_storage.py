"""Tests for rebuilding the tables from object storage.

Runs against real Postgres, not SQLite: the whole tool is an `INSERT ... ON
CONFLICT` upsert, and the conflict semantics are exactly what must hold for a
rerun to be a no-op (NFR-2).

Storage is `LocalStorage` over a tmp dir, which exercises the same `list_keys`
contract the GCS backend implements — so the reconciler is tested end to end
without touching a bucket.
"""

from __future__ import annotations

import pytest
from sqlalchemy import Engine, func, select, text

from app import db
from app.artifact_keys import (
    NUM_VIEWS,
    converted_key,
    normalized_key,
    raw_key,
    renders_prefix,
    view_key,
)
from app.models import Artifact, ArtifactStage, ArtifactStatus, DownloadStatus, Model
from app.reconcile_from_storage import reconcile
from app.storage import LocalStorage

FULLY_PROCESSED = "uid-complete"
RENDER_INTERRUPTED = "uid-partial-render"
DOWNLOAD_ONLY = "uid-raw-only"


@pytest.fixture
def storage(tmp_path):
    """A store holding three models at different points through the pipeline."""
    store = LocalStorage(tmp_path)

    # Made it all the way through: raw, both meshes, and a full view set.
    store.put_bytes(raw_key(FULLY_PROCESSED), b"glb")
    store.put_bytes(converted_key(FULLY_PROCESSED), b"ply")
    store.put_bytes(normalized_key(FULLY_PROCESSED), b"ply")
    for view_index in range(NUM_VIEWS):
        store.put_bytes(view_key(FULLY_PROCESSED, view_index), b"png")

    # Render died midway — only some views landed.
    store.put_bytes(raw_key(RENDER_INTERRUPTED), b"glb")
    store.put_bytes(converted_key(RENDER_INTERRUPTED), b"ply")
    store.put_bytes(normalized_key(RENDER_INTERRUPTED), b"ply")
    for view_index in range(NUM_VIEWS - 4):
        store.put_bytes(view_key(RENDER_INTERRUPTED, view_index), b"png")

    # Downloaded but never converted.
    store.put_bytes(raw_key(DOWNLOAD_ONLY), b"glb")

    return store


@pytest.fixture
def clean_db(pg_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> Engine:
    monkeypatch.setattr(db, "get_engine", lambda: pg_engine)
    with pg_engine.begin() as connection:
        connection.execute(text("TRUNCATE label, artifact, model RESTART IDENTITY CASCADE"))
    return pg_engine


def _artifact_uids(session, stage: ArtifactStage) -> set[str]:
    return set(
        session.execute(
            select(Artifact.model_uid).where(Artifact.stage == stage)
        ).scalars()
    )


def test_reconcile_rebuilds_models_and_artifacts(clean_db, storage) -> None:
    counts = reconcile(storage, dry_run=False)

    assert counts["models"] == 3
    assert counts["converted"] == 2
    assert counts["normalized"] == 2
    assert counts["rendered"] == 1

    with db.session_scope() as session:
        assert set(session.execute(select(Model.uid)).scalars()) == {
            FULLY_PROCESSED,
            RENDER_INTERRUPTED,
            DOWNLOAD_ONLY,
        }
        assert _artifact_uids(session, ArtifactStage.rendered) == {FULLY_PROCESSED}
        assert _artifact_uids(session, ArtifactStage.converted) == {
            FULLY_PROCESSED,
            RENDER_INTERRUPTED,
        }


def test_partial_render_set_is_not_recorded_as_done(clean_db, storage) -> None:
    """Recording an incomplete set would permanently hide it from a re-render.

    The stage-skip gate trusts the row, so a `done` row over 8 of 12 views means
    the model never gets rendered again and silently trains on missing views.
    """
    counts = reconcile(storage, dry_run=False)

    assert counts["partial_renders_skipped"] == 1
    with db.session_scope() as session:
        assert RENDER_INTERRUPTED not in _artifact_uids(session, ArtifactStage.rendered)


def test_rows_match_what_the_workers_would_have_written(clean_db, storage) -> None:
    """The rebuilt rows must be the ones the pipeline itself writes, not lookalikes."""
    reconcile(storage, dry_run=False)

    with db.session_scope() as session:
        model = session.get(Model, FULLY_PROCESSED)
        assert model.raw_key == raw_key(FULLY_PROCESSED)
        assert model.download_status == DownloadStatus.downloaded

        rendered = session.execute(
            select(Artifact).where(
                Artifact.model_uid == FULLY_PROCESSED,
                Artifact.stage == ArtifactStage.rendered,
            )
        ).scalar_one()
        # The render stage stores the prefix, not an individual view.
        assert rendered.key == renders_prefix(FULLY_PROCESSED)
        assert rendered.status == ArtifactStatus.done

        converted = session.execute(
            select(Artifact).where(
                Artifact.model_uid == FULLY_PROCESSED,
                Artifact.stage == ArtifactStage.converted,
            )
        ).scalar_one()
        assert converted.key == converted_key(FULLY_PROCESSED)


def test_rerun_is_a_no_op(clean_db, storage) -> None:
    """NFR-2: the second pass must not duplicate rows or change them."""
    reconcile(storage, dry_run=False)
    with db.session_scope() as session:
        first_models = session.execute(select(func.count(Model.uid))).scalar_one()
        first_artifacts = session.execute(select(func.count(Artifact.id))).scalar_one()

    reconcile(storage, dry_run=False)
    with db.session_scope() as session:
        assert session.execute(select(func.count(Model.uid))).scalar_one() == first_models
        assert (
            session.execute(select(func.count(Artifact.id))).scalar_one() == first_artifacts
        )


def test_rerun_preserves_backfilled_metadata_and_hashes(clean_db, storage) -> None:
    """A rebuild must not undo `backfill_metadata` or blank a real content hash.

    Neither is recoverable from storage, so overwriting them would lose data the
    reconciler has no way to restore.
    """
    reconcile(storage, dry_run=False)
    with db.session_scope() as session:
        model = session.get(Model, FULLY_PROCESSED)
        model.title = "A Wooden Chair"
        model.tags = ["chair", "furniture"]
        artifact = session.execute(
            select(Artifact).where(
                Artifact.model_uid == FULLY_PROCESSED,
                Artifact.stage == ArtifactStage.converted,
            )
        ).scalar_one()
        artifact.content_hash = "sha256-from-the-real-ingestion"

    reconcile(storage, dry_run=False)

    with db.session_scope() as session:
        model = session.get(Model, FULLY_PROCESSED)
        assert model.title == "A Wooden Chair"
        assert model.tags == ["chair", "furniture"]
        artifact = session.execute(
            select(Artifact).where(
                Artifact.model_uid == FULLY_PROCESSED,
                Artifact.stage == ArtifactStage.converted,
            )
        ).scalar_one()
        assert artifact.content_hash == "sha256-from-the-real-ingestion"


def test_dry_run_writes_nothing(clean_db, storage) -> None:
    counts = reconcile(storage, dry_run=True)

    assert counts["models"] == 3
    with db.session_scope() as session:
        assert session.execute(select(func.count(Model.uid))).scalar_one() == 0


def test_stray_objects_are_counted_not_imported(clean_db, storage, tmp_path) -> None:
    """A key that isn't a pipeline artifact is reported, never turned into a model."""
    storage.put_bytes("raw/README.txt", b"not a mesh")

    counts = reconcile(storage, dry_run=False)

    assert counts["unrecognised_keys"] == 1
    assert counts["models"] == 3


def test_model_row_is_created_for_processed_output_with_no_raw_mesh(
    clean_db, tmp_path
) -> None:
    """Raw is deleted for excluded models (cost guardrail), but the outputs remain.

    The labeling UI reads the processed artifacts, so such a model still needs a
    row — flagged as not-downloaded rather than claimed as downloaded.
    """
    store = LocalStorage(tmp_path)
    store.put_bytes(converted_key("uid-raw-deleted"), b"ply")
    store.put_bytes(normalized_key("uid-raw-deleted"), b"ply")

    counts = reconcile(store, dry_run=False)

    assert counts["processed_without_raw"] == 1
    with db.session_scope() as session:
        model = session.get(Model, "uid-raw-deleted")
        assert model.download_status == DownloadStatus.pending
        assert model.raw_key is None


def test_reconcile_preserves_a_soft_delete(clean_db, storage) -> None:
    """A soft-deleted model must not be resurrected by a rebuild from storage.

    The blobs still exist, so the reconciler sees the model — but `deleted_at`
    lives only in the DB, and the upsert writes only storage-authoritative
    columns, so the deletion has to survive. Without that, deleting a model and
    then reconciling would silently bring it back.
    """
    from datetime import UTC, datetime

    reconcile(storage, dry_run=False)
    with db.session_scope() as session:
        session.get(Model, FULLY_PROCESSED).deleted_at = datetime.now(UTC)

    reconcile(storage, dry_run=False)

    with db.session_scope() as session:
        assert session.get(Model, FULLY_PROCESSED).deleted_at is not None
