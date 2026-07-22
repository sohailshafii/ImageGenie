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

# The four key families. Named separately from the per-uid builders below because
# `app.reconcile_from_storage` lists by family to rebuild the tables from storage.
RAW_PREFIX = "raw/"
CONVERTED_PREFIX = "processed/converted/"
NORMALIZED_PREFIX = "processed/normalized/"
RENDERS_PREFIX = "processed/renders/"

MESH_SUFFIX = ".ply"
RAW_SUFFIX = ".glb"


def raw_key(uid: str) -> str:
    """The downloaded source mesh."""
    return f"{RAW_PREFIX}{uid}{RAW_SUFFIX}"


def converted_key(uid: str) -> str:
    """Convert stage output — the pipeline's canonical PLY."""
    return f"{CONVERTED_PREFIX}{uid}{MESH_SUFFIX}"


def normalized_key(uid: str) -> str:
    """Normalize stage output — centered + unit-scaled PLY. What the viewer loads."""
    return f"{NORMALIZED_PREFIX}{uid}{MESH_SUFFIX}"


def renders_prefix(uid: str) -> str:
    """Prefix under which a model's per-view PNGs live."""
    return f"{RENDERS_PREFIX}{uid}/"


def view_key(uid: str, view_index: int) -> str:
    """One rendered view, ``view_00.png`` … ``view_11.png``."""
    return f"{renders_prefix(uid)}view_{view_index:02d}.png"


def view_keys(uid: str) -> list[str]:
    """Every view key for a model, in view order."""
    return [view_key(uid, index) for index in range(NUM_VIEWS)]


def uid_from_key(key: str) -> str | None:
    """The model uid a pipeline key belongs to, or None if the key isn't one.

    The inverse of the builders above, and the reason the rows are recoverable
    from object storage at all: every key carries its uid, so a bucket listing is
    enough to rebuild `model` and `artifact` without re-ingesting
    (server.md#migrations). Lives here so the forward and reverse mappings can
    never drift apart.

    Unrecognised keys return None rather than raising — a listing may legitimately
    contain stray objects, and the reconciler reports them instead of failing.
    """
    for prefix, suffix in (
        (RAW_PREFIX, RAW_SUFFIX),
        (CONVERTED_PREFIX, MESH_SUFFIX),
        (NORMALIZED_PREFIX, MESH_SUFFIX),
    ):
        if key.startswith(prefix) and key.endswith(suffix):
            uid = key[len(prefix) : -len(suffix)]
            # A uid is one path segment: `raw/a/b.glb` is not a raw mesh key.
            return uid if uid and "/" not in uid else None

    if key.startswith(RENDERS_PREFIX):
        # `processed/renders/<uid>/view_NN.png` — the uid is the segment after
        # the prefix, so a per-view key maps back to its model.
        remainder = key[len(RENDERS_PREFIX) :]
        uid, separator, view = remainder.partition("/")
        return uid if uid and separator and view else None

    return None
