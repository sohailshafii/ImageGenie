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
