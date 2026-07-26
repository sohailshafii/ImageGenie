"""Local CPU smoke for the real training loop (M6 B5, ml.md#training).

Seeds a small, class-SEPARABLE dataset — each class is a distinct color, so the
model can actually learn — as real PNG renders in a temporary LocalStorage root
plus the matching model / artifact / label rows, then runs the real train.py flow
(load → split → snapshot → create_run → run_training → finalize) with a small,
CPU-only Config. It asserts the loss falls and the run + curve + checkpoint land,
then deletes everything it created, so it is safe to re-run against a dev DB.

This is a check of the *training code*, not the render pipeline (tested
separately): it proves the loop learns and the bookkeeping/checkpoint work,
without a GPU or the real renders.

Needs a reachable Postgres with the schema applied (``IMAGEGENIE_DATABASE_URL``,
default ``localhost:5432``). Run with ``make smoke-train``.
"""

import io
import os
import random
import shutil
import tempfile
from pathlib import Path

_STORAGE_ROOT = Path(tempfile.mkdtemp(prefix="imagegenie-smoke-"))
os.environ["IMAGEGENIE_STORAGE_BACKEND"] = "local"
os.environ["IMAGEGENIE_STORAGE_ROOT"] = str(_STORAGE_ROOT)

import numpy as np  # noqa: E402  — env must be set before app.config reads it
from PIL import Image  # noqa: E402
from splits import stratified_split  # noqa: E402
from sqlalchemy import delete, select  # noqa: E402
from taxonomy import ROSTER  # noqa: E402
from train import (  # noqa: E402
    Config,
    create_run,
    data_snapshot,
    finalize_run,
    load_trainable_samples,
    run_training,
)

from app.artifact_keys import renders_prefix, view_keys  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import session_scope  # noqa: E402
from app.models import (  # noqa: E402
    Artifact,
    ArtifactStage,
    ArtifactStatus,
    DownloadStatus,
    Label,
    LabelSource,
    Model,
    TrainingMetric,
    TrainingRun,
    TrainingStatus,
)
from app.storage import build_storage  # noqa: E402

CLASSES = list(ROSTER[:6])
PER_CLASS = 14
VIEW_SIZE = 64  # smaller than the real 224² renders — this smoke only tests the loop
PALETTE = {
    CLASSES[0]: (220, 40, 40),
    CLASSES[1]: (40, 200, 60),
    CLASSES[2]: (40, 60, 220),
    CLASSES[3]: (230, 200, 40),
    CLASSES[4]: (200, 40, 210),
    CLASSES[5]: (40, 210, 210),
}
_noise = random.Random(0)


def _make_png(base: tuple[int, int, int]) -> bytes:
    """A solid class color plus small per-view noise, so views differ but the
    class stays linearly separable — enough for the loop to demonstrably learn."""
    array = np.zeros((VIEW_SIZE, VIEW_SIZE, 3), dtype=np.uint8)
    for channel in range(3):
        array[:, :, channel] = min(255, max(0, base[channel] + _noise.randint(-15, 15)))
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _seed(storage) -> list[str]:
    """Write renders + model/artifact/label rows for the smoke set; return uids."""
    uids: list[str] = []
    for class_name in CLASSES:
        for index in range(PER_CLASS):
            uid = f"smoke-{class_name}-{index}"
            uids.append(uid)
            for key in view_keys(uid):
                storage.put_bytes(key, _make_png(PALETTE[class_name]))
            with session_scope() as session:
                session.add(Model(uid=uid, download_status=DownloadStatus.downloaded))
                session.flush()  # model row before its FK children
                session.add(
                    Artifact(
                        model_uid=uid,
                        stage=ArtifactStage.rendered,
                        status=ArtifactStatus.done,
                        key=renders_prefix(uid),
                    )
                )
                session.add(
                    Label(model_uid=uid, class_name=class_name, source=LabelSource.weak)
                )
    return uids


def _cleanup(uids: list[str], run_id: int | None) -> None:
    with session_scope() as session:
        if run_id is not None:
            session.execute(delete(TrainingMetric).where(TrainingMetric.run_id == run_id))
            session.execute(delete(TrainingRun).where(TrainingRun.id == run_id))
        session.execute(delete(Label).where(Label.model_uid.in_(uids)))
        session.execute(delete(Artifact).where(Artifact.model_uid.in_(uids)))
        session.execute(delete(Model).where(Model.uid.in_(uids)))
    shutil.rmtree(_STORAGE_ROOT, ignore_errors=True)


def main() -> None:
    storage = build_storage(get_settings())
    uids = _seed(storage)
    run_id: int | None = None
    try:
        config = Config(epochs=6, batch_size=16, pretrained=False, device="cpu")
        samples = load_trainable_samples()
        split = stratified_split(samples, config.seed)
        snapshot = data_snapshot(samples, split)
        run_id = create_run(config, snapshot, notes="local cpu smoke")
        print(f"trainable={len(samples)} splits={snapshot['splits']}")
        weights_uri = run_training(config, run_id, split)
        finalize_run(run_id, TrainingStatus.completed, weights_uri=weights_uri)

        with session_scope() as session:
            run = session.get(TrainingRun, run_id)
            losses = [
                point.loss
                for point in session.execute(
                    select(TrainingMetric)
                    .where(TrainingMetric.run_id == run_id)
                    .order_by(TrainingMetric.step)
                ).scalars()
            ]
            assert run.status is TrainingStatus.completed
            assert losses[-1] < losses[0], f"loss did not fall: {losses[0]} -> {losses[-1]}"
            assert storage.exists(weights_uri), "checkpoint blob missing"
        print(
            f"SMOKE PASS: loss {losses[0]:.3f} -> {losses[-1]:.3f}, "
            f"weights at {weights_uri}"
        )
    finally:
        _cleanup(uids, run_id)


if __name__ == "__main__":
    main()
