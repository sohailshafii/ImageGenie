"""Convert worker (FR-2) — preprocessing stage 1 of 3.

Reads a model's raw mesh from storage, flattens it to a single geometry, and
re-exports it in the pipeline's canonical **PLY** format under
``processed/converted/<uid>.ply``. Records an ``artifact`` row and hands the model
to the normalize stage.

**Idempotent (NFR-2):** a redelivered job whose converted artifact already exists
(DB row ``done`` + blob present) re-does no work; the artifact write is an upsert
keyed on ``(model_uid, stage)``. The next stage is enqueued regardless, so a model
that stalled mid-pipeline still makes forward progress (normalize skips if done).

**The textured variant** (``{"uid": ..., "variant": "textured"}``) exports **GLB**
to ``processed/converted_textured/<uid>.glb`` instead, because PLY cannot carry a
texture image and the whole point of that arm is to keep one
(ml.md#the-texture-census-step-0). It writes **no artifact row** — the `artifact`
table is unique on ``(model_uid, stage)``, so a row here would overwrite the one
the default arm's trainability query depends on — so its idempotency rests on the
blob alone.
"""

from __future__ import annotations

import hashlib
import logging

from ..artifact_keys import (
    DEFAULT_VARIANT,
    converted_key,
    file_type_for_raw_key,
    raw_key,
)
from ..config import get_settings
from ..consumer import run_stage
from ..db import session_scope
from ..models import ArtifactStage, Model
from ..queue import publish_next
from ..storage import build_storage
from .artifacts import artifact_done, record_artifact
from .mesh import (
    MAX_TEXTURE_DIMENSION,
    export_glb,
    export_ply,
    load_mesh,
    texture_atlas_size,
)

logger = logging.getLogger(__name__)
STAGE = ArtifactStage.converted


def _export(mesh, variant: str, uid: str) -> bytes:
    """Serialize `mesh` in the format `variant` stores, refusing an unusable atlas.

    The textured arm's texture is whatever `load_mesh`'s concatenate packed, and
    that atlas *grows* with the number of source textures rather than downsampling.
    A model whose atlas exceeds what a renderer will accept has to fail here, where
    it dead-letters visibly, rather than at render time — where the likely outcome
    is a silently untextured image, i.e. a treatment-arm model that is really a
    control-arm model.
    """
    if variant == DEFAULT_VARIANT:
        return export_ply(mesh)
    width, height = texture_atlas_size(mesh)
    if max(width, height) > MAX_TEXTURE_DIMENSION:
        raise ValueError(
            f"{uid}: packed texture atlas is {width}x{height}, over the "
            f"{MAX_TEXTURE_DIMENSION}px limit a renderer can be assumed to accept"
        )
    return export_glb(mesh)


def process(job: dict) -> str:
    """Convert one model to its variant's format. ``"converted"`` or ``"skipped"``."""
    uid = job["uid"]
    variant = job.get("variant", DEFAULT_VARIANT)
    settings = get_settings()
    storage = build_storage(settings)
    output_key = converted_key(uid, variant)

    with session_scope() as session:
        # A variant owns no artifact row (see the module docstring), so its only
        # evidence of a previous run is the blob itself.
        already_done = (
            artifact_done(session, uid, STAGE, storage, output_key)
            if variant == DEFAULT_VARIANT
            else storage.exists(output_key)
        )
        # The source format isn't fixed: ingestion writes GLB, but an admin upload
        # may be STL or OBJ, so the stored key is what says which. Falling back to
        # the default keeps rows written before uploads existed working unchanged.
        model = session.get(Model, uid)
        source_key = (model.raw_key if model else None) or raw_key(uid)

    if already_done:
        logger.info(
            "skip already-converted",
            extra={"uid": uid, "stage": STAGE.value, "variant": variant},
        )
        result = "skipped"
    else:
        mesh = load_mesh(
            storage.get_bytes(source_key), file_type=file_type_for_raw_key(source_key)
        )
        mesh_bytes = _export(mesh, variant, uid)
        content_hash = hashlib.sha256(mesh_bytes).hexdigest()
        storage.put_bytes(output_key, mesh_bytes)
        if variant == DEFAULT_VARIANT:
            with session_scope() as session:
                record_artifact(session, uid, STAGE, output_key, content_hash)
        logger.info(
            "converted",
            extra={
                "uid": uid,
                "stage": STAGE.value,
                "variant": variant,
                "content_hash": content_hash,
            },
        )
        result = "converted"

    publish_next(settings.normalize_topic, uid, variant)
    return result


def main() -> None:
    settings = get_settings()
    run_stage(settings.convert_subscription, settings.convert_topic, process)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
