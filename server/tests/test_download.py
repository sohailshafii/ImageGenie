from pathlib import Path

import pytest
from sqlalchemy import Engine, text

from app import config, db
from app.models import DownloadStatus, Model
from app.workers import download


@pytest.fixture
def download_env(pg_engine: Engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the worker at the test Postgres + a temp storage root; clean tables."""
    monkeypatch.setattr(db, "get_engine", lambda: pg_engine)
    monkeypatch.setattr(download, "get_settings", lambda: config.Settings(storage_root=tmp_path))
    with pg_engine.begin() as connection:
        connection.execute(text("TRUNCATE artifact, model RESTART IDENTITY CASCADE"))
    return tmp_path


def test_download_is_idempotent(
    download_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path = download_env
    source_glb = tmp_path / "src.glb"
    source_glb.write_bytes(b"MESH-BYTES")

    fetch_calls: list[list[str]] = []

    def fake_load_objects(uids: list[str], **kwargs: object) -> dict[str, str]:
        fetch_calls.append(list(uids))
        return {uids[0]: str(source_glb)}

    monkeypatch.setattr(download.objaverse, "load_objects", fake_load_objects)

    publish_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        download, "publish_next", lambda topic, uid: publish_calls.append((topic, uid))
    )

    job = {"uid": "abc123"}
    assert download.process(job) == "downloaded"
    assert download.process(job) == "skipped"  # redelivery must not rework
    assert fetch_calls == [["abc123"]]  # fetched exactly once
    # Both outcomes hand the model to the convert stage (forward progress).
    convert_topic = config.Settings().convert_topic
    assert publish_calls == [(convert_topic, "abc123"), (convert_topic, "abc123")]

    with db.session_scope() as session:
        row = session.get(Model, "abc123")
        assert row is not None
        assert row.download_status == DownloadStatus.downloaded
        assert row.raw_key == "raw/abc123.glb"
        assert row.content_hash  # sha256 recorded

    assert (tmp_path / "raw" / "abc123.glb").read_bytes() == b"MESH-BYTES"
