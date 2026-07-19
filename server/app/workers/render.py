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
# Render shape, not colour: a neutral matte material so surface *form* (not the
# model's own textures/vertex colours, which vary arbitrarily) is what the CNN sees.
# Mid-grey (not near-white) keeps the model from washing out against the light bg.
MATERIAL_BASE_COLOR = [0.55, 0.55, 0.58, 1.0]


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


def _light_offset(yaw_degrees: float, pitch_degrees: float) -> np.ndarray:
    """Rotation applied to the camera pose to aim a light off the view axis.

    A light sharing the camera's pose points straight down the view direction —
    head-on lighting that flattens the model to a silhouette. Composing the camera
    pose with this yaw+pitch rotation aims the light from the side/above instead,
    so shading reveals surface form. Attaching it to the *camera* keeps the lighting
    identical across all orbit angles (consistent per-view shading for the CNN).
    """
    yaw = math.radians(yaw_degrees)
    pitch = math.radians(pitch_degrees)
    rotate_yaw = np.array([
        [math.cos(yaw), 0.0, math.sin(yaw), 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-math.sin(yaw), 0.0, math.cos(yaw), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    rotate_pitch = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, math.cos(pitch), -math.sin(pitch), 0.0],
        [0.0, math.sin(pitch), math.cos(pitch), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    return rotate_yaw @ rotate_pitch


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

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0, roughnessFactor=0.85, baseColorFactor=MATERIAL_BASE_COLOR
    )
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0], ambient_light=[0.3, 0.3, 0.3])
    scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False))
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    camera_node = scene.add(camera, pose=poses[0])

    # Key + fill lights aimed off the view axis (see _light_offset), attached to the
    # camera so shading reveals form and stays consistent across every orbit angle.
    key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
    key_offset = _light_offset(yaw_degrees=-35.0, pitch_degrees=-25.0)  # upper-left
    fill_offset = _light_offset(yaw_degrees=40.0, pitch_degrees=10.0)  # softer, right
    key_node = scene.add(key_light, pose=poses[0] @ key_offset)
    fill_node = scene.add(fill_light, pose=poses[0] @ fill_offset)

    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    try:
        images = []
        for pose in poses:
            scene.set_pose(camera_node, pose)
            scene.set_pose(key_node, pose @ key_offset)
            scene.set_pose(fill_node, pose @ fill_offset)
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
