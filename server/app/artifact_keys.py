"""Object-storage key layout for pipeline artifacts (server.md#object-storage).

The single definition of where each stage's output lives. It sits at app level,
not under ``workers/``, because the API serves these same blobs to the labeling UI
— and a key format duplicated between the writer and the reader is exactly the
kind of drift that fails silently, as a missing image rather than an error.

Deliberately free of heavy imports (no trimesh/pyrender), so the API can import it
without pulling in the render stage's GL stack.
"""

from __future__ import annotations

# Views per model, evenly spaced on a tilted ring (ml/ml.md — the multi-view CNN's
# input). The API relies on this to enumerate a model's renders.
NUM_VIEWS = 12


def raw_key(uid: str) -> str:
    """The downloaded source mesh."""
    return f"raw/{uid}.glb"


def converted_key(uid: str) -> str:
    """Convert stage output — the pipeline's canonical PLY."""
    return f"processed/converted/{uid}.ply"


def normalized_key(uid: str) -> str:
    """Normalize stage output — centered + unit-scaled PLY. What the viewer loads."""
    return f"processed/normalized/{uid}.ply"


def renders_prefix(uid: str) -> str:
    """Prefix under which a model's per-view PNGs live."""
    return f"processed/renders/{uid}/"


def view_key(uid: str, view_index: int) -> str:
    """One rendered view, ``view_00.png`` … ``view_11.png``."""
    return f"{renders_prefix(uid)}view_{view_index:02d}.png"


def view_keys(uid: str) -> list[str]:
    """Every view key for a model, in view order."""
    return [view_key(uid, index) for index in range(NUM_VIEWS)]
