"""Object-storage key layout for pipeline artifacts (server.md#object-storage).

The single definition of where each stage's output lives. It sits at app level,
not under ``workers/``, because the API serves these same blobs to the labeling UI
— and a key format duplicated between the writer and the reader is exactly the
kind of drift that fails silently, as a missing image rather than an error.

Deliberately free of heavy imports (no trimesh/pyrender), so the API can import it
without pulling in the render stage's GL stack.
"""

from __future__ import annotations

from dataclasses import dataclass

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

# --- Artifact variants -------------------------------------------------------
#
# A *variant* is a parallel namespace for the same models processed a different
# way: `processed/renders_textured/<uid>/` alongside `processed/renders/<uid>/`.
# It exists for the texture A/B (ml.md#the-texture-ab), which needs to re-render
# a subset with materials preserved **without destroying the shape-only renders
# it is being compared against** — `view_keys(uid)` gives a model exactly one
# render path, so re-rendering in place would overwrite the control arm and make
# the comparison unreproducible.
#
# The two arms differ in mesh format as well as path: the default pipeline's
# canonical PLY carries no UVs at all, which is why the textured arm keeps GLB
# through convert and normalize.
#
# **Variant keys are deliberately outside the four families above.** A prefix
# listing of `processed/converted/` does not match `processed/converted_textured/`,
# so `app.reconcile_from_storage` never sees them — and it must not, because the
# `artifact` table is unique on `(model_uid, stage)` and a row rebuilt from a
# textured blob would overwrite the row the control arm's trainability query
# depends on. `uid_from_key` stays variant-blind for the same reason.
DEFAULT_VARIANT = "default"
TEXTURED_VARIANT = "textured"


@dataclass(frozen=True)
class VariantLayout:
    """How one variant names its blobs: a path suffix and a mesh format."""

    path_suffix: str
    mesh_suffix: str


VARIANT_TO_LAYOUT = {
    DEFAULT_VARIANT: VariantLayout(path_suffix="", mesh_suffix=MESH_SUFFIX),
    TEXTURED_VARIANT: VariantLayout(path_suffix="_textured", mesh_suffix=".glb"),
}


def layout_for(variant: str) -> VariantLayout:
    """The layout for `variant`, or ``ValueError`` for an unknown one.

    Refusing rather than defaulting matters: a typo'd variant that silently fell
    back to the default paths would have one arm of the A/B quietly writing over
    the other's artifacts, which is the single failure this whole namespace
    exists to prevent.
    """
    try:
        return VARIANT_TO_LAYOUT[variant]
    except KeyError:
        raise ValueError(
            f"unknown artifact variant {variant!r}; "
            f"expected one of {sorted(VARIANT_TO_LAYOUT)}"
        ) from None

# Source-mesh formats the pipeline accepts, mapped to the `file_type` trimesh
# loads them as. Ingestion (Objaverse) only ever produces GLB; the others exist
# for admin upload (web.md#data-upload).
#
# **FBX is deliberately absent** — trimesh has no FBX loader, so it is rejected at
# upload with a clear error rather than failing deep in the convert stage. Adding
# it later means an assimp package in the worker image and an entry here; nothing
# else in the pipeline assumes a format (server.md#data-upload).
RAW_SUFFIX_TO_FILE_TYPE = {
    ".glb": "glb",
    ".stl": "stl",
    ".obj": "obj",
}
# What the download worker writes. Objaverse serves GLB.
DEFAULT_RAW_SUFFIX = ".glb"


def raw_key(uid: str, suffix: str = DEFAULT_RAW_SUFFIX) -> str:
    """The source mesh. `suffix` carries the format, since it is not always GLB."""
    return f"{RAW_PREFIX}{uid}{suffix}"


def file_type_for_raw_key(key: str) -> str:
    """The trimesh `file_type` for a raw key, from its extension.

    Raises ``ValueError`` for anything unsupported: a stage that cannot tell what
    it is holding should fail loudly rather than guess a format and mangle the
    mesh.
    """
    for suffix, file_type in RAW_SUFFIX_TO_FILE_TYPE.items():
        if key.endswith(suffix):
            return file_type
    raise ValueError(f"no supported mesh format for raw key {key!r}")


def converted_key(uid: str, variant: str = DEFAULT_VARIANT) -> str:
    """Convert stage output — the pipeline's canonical PLY, or a variant's format."""
    layout = layout_for(variant)
    return f"processed/converted{layout.path_suffix}/{uid}{layout.mesh_suffix}"


def normalized_key(uid: str, variant: str = DEFAULT_VARIANT) -> str:
    """Normalize stage output — centered + unit-scaled. What the viewer loads."""
    layout = layout_for(variant)
    return f"processed/normalized{layout.path_suffix}/{uid}{layout.mesh_suffix}"


def renders_prefix(uid: str, variant: str = DEFAULT_VARIANT) -> str:
    """Prefix under which a model's per-view PNGs live."""
    return f"processed/renders{layout_for(variant).path_suffix}/{uid}/"


# Training output, not a pipeline artifact — kept out of the "families" above so
# `reconcile_from_storage` (which rebuilds model/artifact rows) never lists it.
WEIGHTS_PREFIX = "processed/models/"


def weights_key(run_id: int) -> str:
    """Saved model weights for a training run — a processed-bucket blob. The DB's
    ``training_run.weights_uri`` stores this key (server.md#object-storage)."""
    return f"{WEIGHTS_PREFIX}{run_id}.pt"


# A dev-set selection, likewise not a pipeline artifact. The LVIS set is a list of
# uids and their gold classes, and it lives in a file rather than the `label`
# table on purpose (ml/build_dev_set.py): a labeled model is a trainable one, so
# storing these as labels would let the second dev set leak into a training run.
# Putting a copy in the bucket keeps that property — it is still not a label — and
# makes the file reachable from a Vertex job, which has no checkout.
DEV_SET_PREFIX = "processed/devsets/"


def dev_set_key(name: str) -> str:
    """The stored copy of a dev-set selection, e.g. ``lvis`` -> the LVIS gold set."""
    return f"{DEV_SET_PREFIX}{name}.csv"


def view_key(uid: str, view_index: int, variant: str = DEFAULT_VARIANT) -> str:
    """One rendered view, ``view_00.png`` … ``view_11.png``."""
    return f"{renders_prefix(uid, variant)}view_{view_index:02d}.png"


def view_keys(uid: str, variant: str = DEFAULT_VARIANT) -> list[str]:
    """Every view key for a model, in view order."""
    return [view_key(uid, index, variant) for index in range(NUM_VIEWS)]


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
    candidate_suffixes = (
        [(RAW_PREFIX, suffix) for suffix in RAW_SUFFIX_TO_FILE_TYPE]
        + [(CONVERTED_PREFIX, MESH_SUFFIX), (NORMALIZED_PREFIX, MESH_SUFFIX)]
    )
    for prefix, suffix in candidate_suffixes:
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
