from pathlib import Path

import numpy as np
import pytest
import trimesh
from sqlalchemy import Engine, select, text

from app import config, db
from app.models import Artifact, ArtifactStage, ArtifactStatus, DownloadStatus, Model
from app.workers import normalize
from app.workers.mesh import export_ply, load_mesh


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

    publish_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        normalize, "publish_next", lambda topic, u: publish_calls.append((topic, u))
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
    assert publish_calls == [(render_topic, uid), (render_topic, uid)]
