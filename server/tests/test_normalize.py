from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image
from sqlalchemy import Engine, select, text

from app import config, db
from app.artifact_keys import DEFAULT_VARIANT, TEXTURED_VARIANT
from app.models import Artifact, ArtifactStage, ArtifactStatus, DownloadStatus, Model
from app.workers import normalize
from app.workers.mesh import export_glb, export_ply, load_mesh, texture_atlas_size


@pytest.fixture
def normalize_env(pg_engine: Engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(db, "get_engine", lambda: pg_engine)
    monkeypatch.setattr(normalize, "get_settings", lambda: config.Settings(storage_root=tmp_path))
    with pg_engine.begin() as connection:
        connection.execute(text("TRUNCATE artifact, model RESTART IDENTITY CASCADE"))
    return tmp_path


def test_normalize_centers_and_unit_scales(
    normalize_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path = normalize_env
    uid = "abc123"

    with db.session_scope() as session:
        session.add(Model(uid=uid, download_status=DownloadStatus.downloaded))

    # A converted mesh that is off-center and much larger than a unit cube.
    off_center_box = trimesh.creation.box(extents=(4.0, 2.0, 6.0))
    off_center_box.apply_translation([10.0, -5.0, 3.0])
    converted_dir = tmp_path / "processed" / "converted"
    converted_dir.mkdir(parents=True)
    (converted_dir / f"{uid}.ply").write_bytes(export_ply(off_center_box))

    publish_calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        normalize,
        "publish_next",
        lambda topic, model_uid, variant=DEFAULT_VARIANT: publish_calls.append(
            (topic, model_uid, variant)
        ),
    )

    job = {"uid": uid}
    assert normalize.process(job) == "normalized"
    assert normalize.process(job) == "skipped"

    normalized = load_mesh(
        (tmp_path / "processed" / "normalized" / f"{uid}.ply").read_bytes(), file_type="ply"
    )
    # Centered on the origin and scaled so the largest extent is 1 (unit cube).
    assert np.allclose(normalized.bounds.mean(axis=0), 0.0, atol=1e-6)
    assert normalized.extents.max() == pytest.approx(1.0, abs=1e-6)

    with db.session_scope() as session:
        rows = session.execute(select(Artifact).where(Artifact.model_uid == uid)).scalars().all()
        assert len(rows) == 1
        assert rows[0].stage == ArtifactStage.normalized
        assert rows[0].status == ArtifactStatus.done

    render_topic = config.Settings().render_topic
    assert publish_calls == [
        (render_topic, uid, DEFAULT_VARIANT),
        (render_topic, uid, DEFAULT_VARIANT),
    ]


def test_textured_variant_normalizes_glb_and_keeps_its_texture(
    normalize_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Centering and scaling touch vertices, so the packed texture must ride through."""
    tmp_path = normalize_env
    uid = "textured1"

    with db.session_scope() as session:
        session.add(Model(uid=uid, download_status=DownloadStatus.downloaded))

    off_center_box = trimesh.creation.box(extents=(4.0, 2.0, 6.0))
    off_center_box.apply_translation([10.0, -5.0, 3.0])
    off_center_box.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(off_center_box.vertices), 2)),
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new("RGB", (8, 8), (30, 160, 90))
        ),
    )
    converted_dir = tmp_path / "processed" / "converted_textured"
    converted_dir.mkdir(parents=True)
    (converted_dir / f"{uid}.glb").write_bytes(export_glb(off_center_box))

    publish_calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        normalize,
        "publish_next",
        lambda topic, model_uid, variant=DEFAULT_VARIANT: publish_calls.append(
            (topic, model_uid, variant)
        ),
    )

    job = {"uid": uid, "variant": TEXTURED_VARIANT}
    assert normalize.process(job) == "normalized"
    assert normalize.process(job) == "skipped"

    normalized = load_mesh(
        (tmp_path / "processed" / "normalized_textured" / f"{uid}.glb").read_bytes(),
        file_type="glb",
    )
    assert np.allclose(normalized.bounds.mean(axis=0), 0.0, atol=1e-6)
    assert normalized.extents.max() == pytest.approx(1.0, abs=1e-6)
    assert texture_atlas_size(normalized) != (0, 0)

    # No artifact row, and the default arm's normalized blob was never written.
    with db.session_scope() as session:
        rows = session.execute(select(Artifact).where(Artifact.model_uid == uid)).scalars().all()
        assert rows == []
    assert not (tmp_path / "processed" / "normalized" / f"{uid}.ply").exists()

    render_topic = config.Settings().render_topic
    assert publish_calls == [
        (render_topic, uid, TEXTURED_VARIANT),
        (render_topic, uid, TEXTURED_VARIANT),
    ]
