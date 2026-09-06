from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image
from sqlalchemy import Engine, select, text

from app import config, db
from app.artifact_keys import DEFAULT_VARIANT, TEXTURED_VARIANT
from app.models import Artifact, ArtifactStage, ArtifactStatus, DownloadStatus, Model
from app.workers import convert
from app.workers.mesh import load_mesh, texture_atlas_size


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

    publish_calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        convert,
        "publish_next",
        lambda topic, model_uid, variant=DEFAULT_VARIANT: publish_calls.append(
            (topic, model_uid, variant)
        ),
    )

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
    assert publish_calls == [
        (normalize_topic, uid, DEFAULT_VARIANT),
        (normalize_topic, uid, DEFAULT_VARIANT),
    ]


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

    monkeypatch.setattr(
        convert, "publish_next", lambda topic, model_uid, variant=DEFAULT_VARIANT: None
    )

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

    monkeypatch.setattr(
        convert, "publish_next", lambda topic, model_uid, variant=DEFAULT_VARIANT: None
    )

    assert convert.process({"uid": uid}) == "converted"


# --- The textured variant ---------------------------------------------------


def _seed_model(tmp_path: Path, uid: str, glb_bytes: bytes) -> None:
    """What the download stage leaves behind: a model row and the raw GLB."""
    with db.session_scope() as session:
        session.add(
            Model(uid=uid, download_status=DownloadStatus.downloaded, raw_key=f"raw/{uid}.glb")
        )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(exist_ok=True)
    (raw_dir / f"{uid}.glb").write_bytes(glb_bytes)


def _textured_box_glb() -> bytes:
    """A box carrying a real base-colour image, so the material has something to lose."""
    box = trimesh.creation.box()
    box.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(box.vertices), 2)),
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new("RGB", (8, 8), (200, 40, 40))
        ),
    )
    return box.export(file_type="glb")


def test_textured_variant_writes_glb_and_no_artifact_row(
    convert_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The arm keeps its material, and must not touch the row the default arm owns."""
    tmp_path = convert_env
    uid = "textured1"
    _seed_model(tmp_path, uid, _textured_box_glb())
    publish_calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        convert,
        "publish_next",
        lambda topic, model_uid, variant=DEFAULT_VARIANT: publish_calls.append(
            (topic, model_uid, variant)
        ),
    )

    job = {"uid": uid, "variant": TEXTURED_VARIANT}
    assert convert.process(job) == "converted"
    assert convert.process(job) == "skipped"  # idempotent on the blob alone

    converted = tmp_path / "processed" / "converted_textured" / f"{uid}.glb"
    assert converted.exists()
    assert not (tmp_path / "processed" / "converted" / f"{uid}.ply").exists()

    # The texture survives the export, which is the entire point of the arm.
    reloaded = load_mesh(converted.read_bytes(), file_type="glb")
    assert texture_atlas_size(reloaded) != (0, 0)

    with db.session_scope() as session:
        rows = session.execute(select(Artifact).where(Artifact.model_uid == uid)).scalars().all()
        assert rows == []

    # The variant rides along to the next stage, or the arms would cross.
    assert publish_calls == [
        (config.Settings().normalize_topic, uid, TEXTURED_VARIANT),
        (config.Settings().normalize_topic, uid, TEXTURED_VARIANT),
    ]


def test_the_two_arms_do_not_share_a_blob(
    convert_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Converting both arms of the same model leaves both outputs intact."""
    tmp_path = convert_env
    uid = "botharms"
    _seed_model(tmp_path, uid, _textured_box_glb())
    monkeypatch.setattr(
        convert, "publish_next", lambda topic, model_uid, variant=DEFAULT_VARIANT: None
    )

    assert convert.process({"uid": uid}) == "converted"
    assert convert.process({"uid": uid, "variant": TEXTURED_VARIANT}) == "converted"

    assert (tmp_path / "processed" / "converted" / f"{uid}.ply").exists()
    assert (tmp_path / "processed" / "converted_textured" / f"{uid}.glb").exists()
    # And the default arm still owns exactly the one row it did before.
    with db.session_scope() as session:
        rows = session.execute(select(Artifact).where(Artifact.model_uid == uid)).scalars().all()
        assert len(rows) == 1
        assert rows[0].key == f"processed/converted/{uid}.ply"


def test_an_oversized_atlas_is_refused_at_convert(
    convert_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail where it dead-letters visibly, not at render where it goes quiet.

    A model whose packed atlas exceeds GL_MAX_TEXTURE_SIZE would most likely render
    untextured — a treatment-arm model that is really a control-arm model, and
    nothing about the resulting number would look wrong.
    """
    tmp_path = convert_env
    uid = "hugeatlas"
    _seed_model(tmp_path, uid, _textured_box_glb())
    monkeypatch.setattr(
        convert, "publish_next", lambda topic, model_uid, variant=DEFAULT_VARIANT: None
    )
    monkeypatch.setattr(convert, "texture_atlas_size", lambda mesh: (32768, 2048))

    with pytest.raises(ValueError, match="packed texture atlas is 32768x2048"):
        convert.process({"uid": uid, "variant": TEXTURED_VARIANT})
    assert not (tmp_path / "processed" / "converted_textured").exists()

    # The default arm is unaffected: PLY carries no atlas, so nothing to refuse.
    assert convert.process({"uid": uid}) == "converted"
