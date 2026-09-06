from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image
from sqlalchemy import Engine, select, text

from app import config, db
from app.artifact_keys import TEXTURED_VARIANT
from app.models import Artifact, ArtifactStage, ArtifactStatus, DownloadStatus, Model
from app.workers import render
from app.workers.mesh import export_glb, export_ply


@pytest.fixture
def render_env(pg_engine: Engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(db, "get_engine", lambda: pg_engine)
    monkeypatch.setattr(render, "get_settings", lambda: config.Settings(storage_root=tmp_path))
    with pg_engine.begin() as connection:
        connection.execute(text("TRUNCATE artifact, model RESTART IDENTITY CASCADE"))
    return tmp_path


def test_camera_poses_are_distinct_transforms() -> None:
    poses = render.camera_poses(render.NUM_VIEWS)
    assert len(poses) == render.NUM_VIEWS
    assert all(pose.shape == (4, 4) for pose in poses)
    # Adjacent viewpoints differ (the ring is not degenerate).
    assert not (poses[0] == poses[1]).all()


def test_render_writes_view_set_idempotently(
    render_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path = render_env
    uid = "abc123"

    with db.session_scope() as session:
        session.add(Model(uid=uid, download_status=DownloadStatus.downloaded))
    normalized_dir = tmp_path / "processed" / "normalized"
    normalized_dir.mkdir(parents=True)
    (normalized_dir / f"{uid}.ply").write_bytes(export_ply(trimesh.creation.box()))

    # Isolate the test from a real GL context: fake one PNG per requested view.
    render_calls: list[int] = []

    def fake_render_views(
        mesh: object, poses: list, resolution: int, use_mesh_material: bool = False
    ) -> list[bytes]:
        render_calls.append(len(poses))
        assert use_mesh_material is False  # the default arm renders shape, not colour
        return [f"png-{index}".encode() for index in range(len(poses))]

    monkeypatch.setattr(render, "render_views", fake_render_views)

    job = {"uid": uid}
    assert render.process(job) == "rendered"
    assert render.process(job) == "skipped"  # full set present → no re-render
    assert render_calls == [render.NUM_VIEWS]  # rendered exactly once

    renders_dir = tmp_path / "processed" / "renders" / uid
    written = sorted(path.name for path in renders_dir.iterdir())
    assert written == [f"view_{index:02d}.png" for index in range(render.NUM_VIEWS)]

    with db.session_scope() as session:
        rows = session.execute(select(Artifact).where(Artifact.model_uid == uid)).scalars().all()
        assert len(rows) == 1
        assert rows[0].stage == ArtifactStage.rendered
        assert rows[0].status == ArtifactStatus.done
        assert rows[0].key == f"processed/renders/{uid}/"  # prefix, not a single file


# --- The textured variant ---------------------------------------------------


def _textured_box() -> trimesh.Trimesh:
    """A box carrying a base-colour image — the material the arm exists to render."""
    box = trimesh.creation.box()
    box.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(box.vertices), 2)),
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new("RGB", (8, 8), (10, 90, 200))
        ),
    )
    return box


def test_textured_variant_renders_the_meshs_own_material(
    render_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one thing the A/B varies, and the only thing this stage may vary."""
    tmp_path = render_env
    uid = "textured1"

    with db.session_scope() as session:
        session.add(Model(uid=uid, download_status=DownloadStatus.downloaded))
    normalized_dir = tmp_path / "processed" / "normalized_textured"
    normalized_dir.mkdir(parents=True)
    (normalized_dir / f"{uid}.glb").write_bytes(export_glb(_textured_box()))

    captured: list[tuple[int, int, bool]] = []

    def fake_render_views(
        mesh: object, poses: list, resolution: int, use_mesh_material: bool = False
    ) -> list[bytes]:
        captured.append((len(poses), resolution, use_mesh_material))
        return [f"png-{index}".encode() for index in range(len(poses))]

    monkeypatch.setattr(render, "render_views", fake_render_views)

    job = {"uid": uid, "variant": TEXTURED_VARIANT}
    assert render.process(job) == "rendered"
    assert render.process(job) == "skipped"

    # Same view count and resolution as the control arm; only the material differs.
    assert captured == [(render.NUM_VIEWS, render.RESOLUTION, True)]

    renders_dir = tmp_path / "processed" / "renders_textured" / uid
    assert sorted(path.name for path in renders_dir.iterdir()) == [
        f"view_{index:02d}.png" for index in range(render.NUM_VIEWS)
    ]
    assert not (tmp_path / "processed" / "renders" / uid).exists()

    with db.session_scope() as session:
        rows = session.execute(select(Artifact).where(Artifact.model_uid == uid)).scalars().all()
        assert rows == []


def test_a_partial_variant_view_set_is_re_rendered(
    render_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no artifact row, the view set is the only record — so it must be checked.

    A crash midway leaves some PNGs behind. Treating those as done would leave a
    model in the treatment arm with fewer views than the dataset expects.
    """
    tmp_path = render_env
    uid = "partial1"

    with db.session_scope() as session:
        session.add(Model(uid=uid, download_status=DownloadStatus.downloaded))
    normalized_dir = tmp_path / "processed" / "normalized_textured"
    normalized_dir.mkdir(parents=True)
    (normalized_dir / f"{uid}.glb").write_bytes(export_glb(_textured_box()))
    partial_dir = tmp_path / "processed" / "renders_textured" / uid
    partial_dir.mkdir(parents=True)
    (partial_dir / "view_00.png").write_bytes(b"png-0")

    monkeypatch.setattr(
        render,
        "render_views",
        lambda mesh, poses, resolution, use_mesh_material=False: [
            f"png-{index}".encode() for index in range(len(poses))
        ],
    )

    assert render.process({"uid": uid, "variant": TEXTURED_VARIANT}) == "rendered"
    assert len(list(partial_dir.iterdir())) == render.NUM_VIEWS
