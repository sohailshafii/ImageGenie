"""Shared mesh helpers for the preprocessing stages (convert / normalize / render).

``trimesh`` is a hard dependency of every preprocessing stage, so it is imported
at module load. Objaverse GLBs can be a multi-geometry **scene**; ``load_mesh``
flattens that to a single ``Trimesh`` so downstream stages work on one geometry.
"""

from __future__ import annotations

import io

import trimesh


def load_mesh(data: bytes, file_type: str) -> trimesh.Trimesh:
    """Load `data` as a single mesh, concatenating any scene geometries.

    Raises ``ValueError`` if the payload has no faces (empty, or points/curves
    only) — a poison mesh the caller lets fail so Pub/Sub dead-letters it after
    the max delivery attempts (server.md#queue), rather than looping forever.
    """
    loaded = trimesh.load(io.BytesIO(data), file_type=file_type)
    if isinstance(loaded, trimesh.Scene):
        geometries = tuple(loaded.geometry.values())
        if not geometries:
            raise ValueError("scene has no geometry")
        mesh = trimesh.util.concatenate(geometries)
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh) or mesh.faces.shape[0] == 0:
        raise ValueError("mesh has no faces")
    return mesh


def export_ply(mesh: trimesh.Trimesh) -> bytes:
    """Serialize `mesh` to binary PLY bytes — the pipeline's canonical format."""
    return mesh.export(file_type="ply")


def export_glb(mesh: trimesh.Trimesh) -> bytes:
    """Serialize `mesh` to binary GLB — the format that can still carry materials.

    PLY has no way to store a texture image (it does store the ``s``/``t``
    coordinates, which is worse than losing them: the reloaded mesh reports UVs
    while its ``baseColorTexture`` is gone). GLB is what the textured variant of
    the pipeline keeps, so the material survives to the render stage
    (ml.md#the-texture-census-step-0).
    """
    return mesh.export(file_type="glb")


# The largest texture dimension a renderer can be assumed to accept. 16384 is the
# usual `GL_MAX_TEXTURE_SIZE`, and `concatenate` packs a multi-geometry model's
# textures into one atlas that grows rather than downsamples: measured over 50
# sampled models the median atlas is 2048 wide, but 4 of 50 pack to exactly 16384.
MAX_TEXTURE_DIMENSION = 16384


def texture_atlas_size(mesh: trimesh.Trimesh) -> tuple[int, int]:
    """``(width, height)`` of the mesh's base-colour image, or ``(0, 0)`` if none.

    After `load_mesh` this is the *packed* atlas: concatenating a multi-geometry
    GLB merges its materials into one image and rewrites the UVs to index it.
    """
    material = getattr(mesh.visual, "material", None)
    image = getattr(material, "baseColorTexture", None)
    return image.size if image is not None else (0, 0)
