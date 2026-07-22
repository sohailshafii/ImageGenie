from pathlib import Path

import pytest
import trimesh
from sqlalchemy import Engine, select, text

from app import config, db
from app.models import Artifact, ArtifactStage, ArtifactStatus, DownloadStatus, Model
from app.workers import convert
from app.workers.mesh import load_mesh


@pytest.fixture
def convert_env(pg_engine: Engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the worker at the test Postgres + a temp storage root; clean tables."""
    monkeypatch.setattr(db, "get_engine", lambda: pg_engine)
    monkeypatch.setattr(convert, "get_settings", lambda: config.Settings(storage_root=tmp_path))
    with pg_engine.begin() as connection:
        connection.execute(text("TRUNCATE artifact, model RESTART IDENTITY CASCADE"))
    return tmp_path


def test_convert_is_idempotent(convert_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path = convert_env
    uid = "abc123"

    # The download stage's outputs: a model row + the raw GLB blob.
    with db.session_scope() as session:
        session.add(
            Model(uid=uid, download_status=DownloadStatus.downloaded, raw_key=f"raw/{uid}.glb")
        )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / f"{uid}.glb").write_bytes(trimesh.creation.box().export(file_type="glb"))

    publish_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(convert, "publish_next", lambda topic, u: publish_calls.append((topic, u)))

    job = {"uid": uid}
    assert convert.process(job) == "converted"
    assert convert.process(job) == "skipped"  # redelivery must not rework

    # The converted blob is a valid PLY with geometry.
    converted_ply = (tmp_path / "processed" / "converted" / f"{uid}.ply").read_bytes()
    assert load_mesh(converted_ply, file_type="ply").faces.shape[0] > 0

    # Exactly one artifact row, marked done, keyed on this uid + stage.
    with db.session_scope() as session:
        rows = session.execute(select(Artifact).where(Artifact.model_uid == uid)).scalars().all()
        assert len(rows) == 1
        assert rows[0].stage == ArtifactStage.converted
        assert rows[0].status == ArtifactStatus.done
        assert rows[0].content_hash  # sha256 recorded

    # Both runs hand the model to the normalize stage.
    normalize_topic = config.Settings().normalize_topic
    assert publish_calls == [(normalize_topic, uid), (normalize_topic, uid)]


@pytest.mark.parametrize(
    ("suffix", "export_type"), [(".stl", "stl"), (".obj", "obj")]
)
def test_convert_reads_the_format_from_the_stored_raw_key(
    convert_env: Path, monkeypatch: pytest.MonkeyPatch, suffix: str, export_type: str
) -> None:
    """An uploaded STL/OBJ converts too — the stage must not assume GLB.

    The format is carried by `model.raw_key`, which upload sets to the extension
    it validated, so nothing downstream has to be told separately.
    """
    tmp_path = convert_env
    uid = f"upload{export_type}"

    with db.session_scope() as session:
        session.add(
            Model(
                uid=uid,
                download_status=DownloadStatus.downloaded,
                raw_key=f"raw/{uid}{suffix}",
            )
        )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(exist_ok=True)
    # trimesh exports the text formats (OBJ) as str and the binary ones as bytes.
    exported = trimesh.creation.box().export(file_type=export_type)
    (raw_dir / f"{uid}{suffix}").write_bytes(
        exported.encode() if isinstance(exported, str) else exported
    )

    monkeypatch.setattr(convert, "publish_next", lambda topic, u: None)

    assert convert.process({"uid": uid}) == "converted"

    converted_ply = (tmp_path / "processed" / "converted" / f"{uid}.ply").read_bytes()
    assert load_mesh(converted_ply, file_type="ply").faces.shape[0] > 0


def test_convert_falls_back_to_glb_when_no_raw_key_is_recorded(
    convert_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows predating uploads have raw_key set, but a null one must still work."""
    tmp_path = convert_env
    uid = "no-raw-key"

    with db.session_scope() as session:
        session.add(Model(uid=uid, download_status=DownloadStatus.downloaded))
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(exist_ok=True)
    (raw_dir / f"{uid}.glb").write_bytes(trimesh.creation.box().export(file_type="glb"))

    monkeypatch.setattr(convert, "publish_next", lambda topic, u: None)

    assert convert.process({"uid": uid}) == "converted"
