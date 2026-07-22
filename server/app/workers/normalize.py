"""Normalize worker (FR-2) — preprocessing stage 2 of 3.

Reads a model's converted PLY, **centers** it on its bounding-box center and
**rescales** it so its largest extent is 1 (a unit cube), then validates the
result and writes ``processed/normalized/<uid>.ply``. Centering + unit scaling
make renders framing-invariant across wildly different source sizes, which the
downstream multi-view CNN depends on. Records an ``artifact`` row and hands the
model to the render stage.

**Idempotent (NFR-2):** a redelivered job whose normalized artifact already exists
is skipped; the artifact write is an upsert keyed on ``(model_uid, stage)``.
"""

from __future__ import annotations

import hashlib
import logging

from ..artifact_keys import converted_key, normalized_key
from ..config import get_settings
from ..consumer import run_stage
from ..db import session_scope
from ..models import ArtifactStage
from ..queue import publish_next
from ..storage import build_storage
from .artifacts import artifact_done, record_artifact
from .mesh import export_ply, load_mesh

logger = logging.getLogger(__name__)
STAGE = ArtifactStage.normalized




def process(job: dict) -> str:
    """Center + unit-scale one model. Returns ``"normalized"`` or ``"skipped"``."""
    uid = job["uid"]
    settings = get_settings()
    storage = build_storage(settings)
    output_key = normalized_key(uid)

    with session_scope() as session:
        already_done = artifact_done(session, uid, STAGE, storage, output_key)

    if already_done:
        logger.info("skip already-normalized", extra={"uid": uid, "stage": STAGE.value})
        result = "skipped"
    else:
        mesh = load_mesh(storage.get_bytes(converted_key(uid)), file_type="ply")

        # Center on the bounding-box center, then scale the largest extent to 1.
        bounding_box_center = mesh.bounds.mean(axis=0)
        mesh.apply_translation(-bounding_box_center)
        largest_extent = float(mesh.extents.max())
        if largest_extent <= 0.0:
            raise ValueError(f"degenerate mesh (zero extent) for {uid}")
        mesh.apply_scale(1.0 / largest_extent)

        ply_bytes = export_ply(mesh)
        content_hash = hashlib.sha256(ply_bytes).hexdigest()
        storage.put_bytes(output_key, ply_bytes)
        with session_scope() as session:
            record_artifact(session, uid, STAGE, output_key, content_hash)
        logger.info(
            "normalized",
            extra={"uid": uid, "stage": STAGE.value, "content_hash": content_hash},
        )
        result = "normalized"

    publish_next(settings.render_topic, uid)
    return result


def main() -> None:
    settings = get_settings()
    run_stage(settings.normalize_subscription, settings.normalize_topic, process)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
