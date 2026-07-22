"""Convert worker (FR-2) — preprocessing stage 1 of 3.

Reads a model's raw mesh (GLB) from storage, flattens it to a single geometry,
and re-exports it in the pipeline's canonical **PLY** format under
``processed/converted/<uid>.ply``. Records an ``artifact`` row and hands the model
to the normalize stage.

**Idempotent (NFR-2):** a redelivered job whose converted artifact already exists
(DB row ``done`` + blob present) re-does no work; the artifact write is an upsert
keyed on ``(model_uid, stage)``. The next stage is enqueued regardless, so a model
that stalled mid-pipeline still makes forward progress (normalize skips if done).
"""

from __future__ import annotations

import hashlib
import logging

from ..artifact_keys import converted_key, raw_key
from ..config import get_settings
from ..consumer import run_stage
from ..db import session_scope
from ..models import ArtifactStage
from ..queue import publish_next
from ..storage import build_storage
from .artifacts import artifact_done, record_artifact
from .mesh import export_ply, load_mesh

logger = logging.getLogger(__name__)
STAGE = ArtifactStage.converted


def process(job: dict) -> str:
    """Convert one model to canonical PLY. Returns ``"converted"`` or ``"skipped"``."""
    uid = job["uid"]
    settings = get_settings()
    storage = build_storage(settings)
    output_key = converted_key(uid)

    with session_scope() as session:
        already_done = artifact_done(session, uid, STAGE, storage, output_key)

    if already_done:
        logger.info("skip already-converted", extra={"uid": uid, "stage": STAGE.value})
        result = "skipped"
    else:
        mesh = load_mesh(storage.get_bytes(raw_key(uid)), file_type="glb")
        ply_bytes = export_ply(mesh)
        content_hash = hashlib.sha256(ply_bytes).hexdigest()
        storage.put_bytes(output_key, ply_bytes)
        with session_scope() as session:
            record_artifact(session, uid, STAGE, output_key, content_hash)
        logger.info(
            "converted",
            extra={"uid": uid, "stage": STAGE.value, "content_hash": content_hash},
        )
        result = "converted"

    publish_next(settings.normalize_topic, uid)
    return result


def main() -> None:
    settings = get_settings()
    run_stage(settings.convert_subscription, settings.convert_topic, process)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
