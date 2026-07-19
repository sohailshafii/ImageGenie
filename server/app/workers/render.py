"""Render worker (FR-2) — preprocessing stage 3 of 3 (terminal).

Reads a model's normalized PLY and renders it from ``NUM_VIEWS`` viewpoints on a
ring around the object, writing one PNG per view under
``processed/renders/<uid>/view_NN.png``. These multi-view images are the training
input for the milestone-6 multi-view CNN (ml/ml.md#representation).

Rendering is **offscreen** via pyrender (OSMesa in the container; see the
Dockerfile). The pure camera-pose math (`_camera_poses`) is separated from the GL
call (`_render_views`) so the geometry is unit-testable without a GL context.

**Idempotent (NFR-2):** a redelivered job whose full view set already exists is
skipped; the artifact write is an upsert keyed on ``(model_uid, stage)``. This is
the last stage, so nothing is enqueued downstream.
"""

from __future__ import annotations

import io
import logging
import math

import numpy as np

from ..config import get_settings
from ..consumer import run_stage
from ..db import session_scope
from ..models import ArtifactStage
from ..storage import Storage, build_storage
from .artifacts import artifact_done, record_artifact
from .mesh import load_mesh

logger = logging.getLogger(__name__)
STAGE = ArtifactStage.rendered

# ~12 views feeding a standard CNN at ResNet's native input size (ml/ml.md).
NUM_VIEWS = 12
RESOLUTION = 224
# The normalized mesh fits a unit cube; this camera distance frames it with margin
# at pyrender's default ~45° vertical FOV, and the ring is tilted up for a 3/4 view.
CAMERA_RADIUS = 2.2
CAMERA_ELEVATION = 0.8


def _normalized_key(uid: str) -> str:
    return f"processed/normalized/{uid}.ply"


def _renders_prefix(uid: str) -> str:
    return f"processed/renders/{uid}/"


def _view_key(uid: str, view_index: int) -> str:
    return f"{_renders_prefix(uid)}view_{view_index:02d}.png"


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Camera-to-world 4x4 pose placing the camera at `eye` looking at `target`.

    Follows pyrender's convention: the camera looks down its local -Z axis.
    """
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def _camera_poses(num_views: int) -> list[np.ndarray]:
    """`num_views` camera-to-world poses evenly spaced on a tilted ring at origin."""
    target = np.zeros(3)
    up = np.array([0.0, 1.0, 0.0])
    poses = []
    for view_index in range(num_views):
        angle = 2.0 * math.pi * view_index / num_views
        eye = np.array([
            CAMERA_RADIUS * math.cos(angle),
            CAMERA_ELEVATION,
            CAMERA_RADIUS * math.sin(angle),
        ])
        poses.append(_look_at(eye, target, up))
    return poses


def _render_views(mesh, poses: list[np.ndarray], resolution: int) -> list[bytes]:
    """Render `mesh` from each pose to PNG bytes (offscreen; imports pyrender lazily).

    pyrender/OpenGL are imported here, not at module load, so the pure geometry
    above (and its tests) run without a GL context or the OSMesa system libs.
    """
    import pyrender
    from PIL import Image

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0], ambient_light=[0.4, 0.4, 0.4])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    camera_node = scene.add(camera, pose=poses[0])
    light_node = scene.add(light, pose=poses[0])

    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    try:
        images = []
        for pose in poses:
            scene.set_pose(camera_node, pose)
            scene.set_pose(light_node, pose)
            color, _ = renderer.render(scene)
            buffer = io.BytesIO()
            Image.fromarray(color).save(buffer, format="PNG")
            images.append(buffer.getvalue())
        return images
    finally:
        renderer.delete()


def _all_views_present(storage: Storage, uid: str) -> bool:
    return all(storage.exists(_view_key(uid, index)) for index in range(NUM_VIEWS))


def process(job: dict) -> str:
    """Render one model's view set. Returns ``"rendered"`` or ``"skipped"``."""
    uid = job["uid"]
    settings = get_settings()
    storage = build_storage(settings)
    renders_prefix = _renders_prefix(uid)

    with session_scope() as session:
        already_done = artifact_done(
            session, uid, STAGE, storage, _view_key(uid, NUM_VIEWS - 1)
        )
    # Guard the last-view marker against a partially-written set from a prior crash.
    if already_done and _all_views_present(storage, uid):
        logger.info("skip already-rendered", extra={"uid": uid, "stage": STAGE.value})
        return "skipped"

    mesh = load_mesh(storage.get_bytes(_normalized_key(uid)), file_type="ply")
    images = _render_views(mesh, _camera_poses(NUM_VIEWS), RESOLUTION)
    for view_index, png_bytes in enumerate(images):
        storage.put_bytes(_view_key(uid, view_index), png_bytes)
    with session_scope() as session:
        record_artifact(session, uid, STAGE, renders_prefix, content_hash=None)
    logger.info(
        "rendered", extra={"uid": uid, "stage": STAGE.value, "views": len(images)}
    )
    return "rendered"


def main() -> None:
    settings = get_settings()
    run_stage(settings.render_subscription, settings.render_topic, process)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
