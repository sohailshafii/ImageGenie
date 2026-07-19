from pathlib import Path

import pytest
import trimesh
from sqlalchemy import Engine, select, text

from app import config, db
from app.models import Artifact, ArtifactStage, ArtifactStatus, DownloadStatus, Model
from app.workers import render
from app.workers.mesh import export_ply


@pytest.fixture
def render_env(pg_engine: Engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(db, "get_engine", lambda: pg_engine)
    monkeypatch.setattr(render, "get_settings", lambda: config.Settings(storage_root=tmp_path))
    with pg_engine.begin() as connection:
        connection.execute(text("TRUNCATE artifact, model RESTART IDENTITY CASCADE"))
    return tmp_path


def test_camera_poses_are_distinct_transforms() -> None:
    poses = render._camera_poses(render.NUM_VIEWS)
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

    def fake_render_views(mesh: object, poses: list, resolution: int) -> list[bytes]:
        render_calls.append(len(poses))
        return [f"png-{index}".encode() for index in range(len(poses))]

    monkeypatch.setattr(render, "_render_views", fake_render_views)

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
