"""Normalize worker (FR-2) — preprocessing stage 2 of 3.

Reads a model's converted PLY, **centers** it on its bounding-box center and
**rescales** it so its largest extent is 1 (a unit cube), then validates the
result and writes ``processed/normalized/<uid>.ply``. Centering + unit scaling
make renders framing-invariant across wildly different source sizes, which the
downstream multi-view CNN depends on. Records an ``artifact`` row and hands the
model to the render stage.

**Idempotent (NFR-2):** a redelivered job whose normalized artifact already exists
is skipped; the artifact write is an upsert keyed on ``(model_uid, stage)``.

**The textured variant** reads and writes GLB under ``processed/normalized_textured/``
and records no artifact row, for the reasons in `convert.py`. The transform itself is
unchanged: centering and unit-scaling touch vertices, not materials, so the packed
texture rides through untouched.
"""

from __future__ import annotations

import hashlib
import logging

import trimesh

from ..artifact_keys import (
    DEFAULT_VARIANT,
    converted_key,
    mesh_file_type,
    normalized_key,
)
from ..config import get_settings
from ..consumer import run_stage
from ..db import session_scope
from ..models import ArtifactStage
from ..queue import publish_next
from ..storage import build_storage
from .artifacts import artifact_done, record_artifact
from .mesh import export_glb, export_ply, load_mesh

logger = logging.getLogger(__name__)
STAGE = ArtifactStage.normalized




def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Center on the bounding-box center and scale the largest extent to 1.

    Its own function because prediction has to apply the *same* transform to an
    uploaded mesh (app/predict.py): a model trained on unit-scale, origin-centred
    renders sees something else entirely if the views come from a mesh at a
    different scale or offset, and the mismatch would surface as a bad prediction
    rather than as an error.
    """
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    largest_extent = float(mesh.extents.max())
    if largest_extent <= 0.0:
        raise ValueError("degenerate mesh (zero extent)")
    mesh.apply_scale(1.0 / largest_extent)
    return mesh


def process(job: dict) -> str:
    """Center + unit-scale one model. Returns ``"normalized"`` or ``"skipped"``."""
    uid = job["uid"]
    variant = job.get("variant", DEFAULT_VARIANT)
    settings = get_settings()
    storage = build_storage(settings)
    output_key = normalized_key(uid, variant)

    with session_scope() as session:
        # A variant owns no artifact row, so the blob is its only evidence.
        already_done = (
            artifact_done(session, uid, STAGE, storage, output_key)
            if variant == DEFAULT_VARIANT
            else storage.exists(output_key)
        )

    if already_done:
        logger.info(
            "skip already-normalized",
            extra={"uid": uid, "stage": STAGE.value, "variant": variant},
        )
        result = "skipped"
    else:
        mesh = normalize_mesh(
            load_mesh(
                storage.get_bytes(converted_key(uid, variant)),
                file_type=mesh_file_type(variant),
            )
        )
        export = export_ply if variant == DEFAULT_VARIANT else export_glb
        mesh_bytes = export(mesh)
        content_hash = hashlib.sha256(mesh_bytes).hexdigest()
        storage.put_bytes(output_key, mesh_bytes)
        if variant == DEFAULT_VARIANT:
            with session_scope() as session:
                record_artifact(session, uid, STAGE, output_key, content_hash)
        logger.info(
            "normalized",
            extra={
                "uid": uid,
                "stage": STAGE.value,
                "variant": variant,
                "content_hash": content_hash,
            },
        )
        result = "normalized"

    publish_next(settings.render_topic, uid, variant)
    return result


def main() -> None:
    settings = get_settings()
    run_stage(settings.normalize_subscription, settings.normalize_topic, process)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
