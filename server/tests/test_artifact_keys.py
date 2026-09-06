"""Tests for the artifact key layout, especially the reverse mapping.

`uid_from_key` is what lets the tables be rebuilt from a bucket listing
(server.md#migrations), so the property that matters most is that it inverts the
builders exactly — a round trip, not a hand-written string.
"""

from __future__ import annotations

import pytest

from app.artifact_keys import (
    CONVERTED_PREFIX,
    DEFAULT_VARIANT,
    NORMALIZED_PREFIX,
    NUM_VIEWS,
    RENDERS_PREFIX,
    TEXTURED_VARIANT,
    converted_key,
    file_type_for_raw_key,
    normalized_key,
    raw_key,
    renders_prefix,
    uid_from_key,
    view_key,
    view_keys,
)

UID = "0002c6eafa154e8bb08ebafb715a8d46"


@pytest.mark.parametrize("build_key", [raw_key, converted_key, normalized_key])
def test_uid_from_key_inverts_each_builder(build_key) -> None:
    assert uid_from_key(build_key(UID)) == UID


def test_uid_from_key_maps_every_view_back_to_its_model() -> None:
    """All NUM_VIEWS PNGs resolve to the one uid — the render stage's key is a prefix."""
    assert {uid_from_key(key) for key in view_keys(UID)} == {UID}
    assert len(view_keys(UID)) == NUM_VIEWS


@pytest.mark.parametrize(
    "key",
    [
        "",
        "raw/",
        "processed/",
        "processed/renders/",
        "raw/not-a-mesh.txt",  # right prefix, wrong suffix
        "processed/converted/nested/uid.ply",  # a uid is one path segment
        "processed/renders/uid-with-no-view/",  # prefix alone isn't a view
        "some/other/thing.ply",
    ],
)
def test_uid_from_key_returns_none_for_non_pipeline_keys(key: str) -> None:
    """Stray objects are reported by the reconciler, not treated as models."""
    assert uid_from_key(key) is None


def test_render_prefix_and_view_are_distinguished() -> None:
    """The bare prefix yields nothing; a view under it yields the uid."""
    assert uid_from_key(f"processed/renders/{UID}/") is None
    assert uid_from_key(view_key(UID, 0)) == UID


def test_raw_key_defaults_to_glb_and_accepts_other_formats() -> None:
    """Ingestion writes GLB; upload may write STL or OBJ."""
    assert raw_key(UID) == f"raw/{UID}.glb"
    assert raw_key(UID, ".stl") == f"raw/{UID}.stl"


@pytest.mark.parametrize("suffix", [".glb", ".stl", ".obj"])
def test_uid_from_key_handles_every_supported_raw_format(suffix: str) -> None:
    assert uid_from_key(raw_key(UID, suffix)) == UID


@pytest.mark.parametrize(
    ("suffix", "expected_file_type"),
    [(".glb", "glb"), (".stl", "stl"), (".obj", "obj")],
)
def test_file_type_for_raw_key(suffix: str, expected_file_type: str) -> None:
    assert file_type_for_raw_key(raw_key(UID, suffix)) == expected_file_type


def test_file_type_for_raw_key_rejects_unsupported_formats() -> None:
    """FBX has no trimesh loader, so a stage must fail loudly rather than guess."""
    with pytest.raises(ValueError, match="no supported mesh format"):
        file_type_for_raw_key(f"raw/{UID}.fbx")


def test_fbx_is_not_a_recognised_key() -> None:
    """Upload rejects FBX up front; nothing downstream should treat one as ingestible."""
    assert uid_from_key(f"raw/{UID}.fbx") is None


# --- Variants ---------------------------------------------------------------
#
# The textured arm of the A/B re-processes models the pipeline has already
# rendered. Every test here is really one property: it must be impossible for one
# arm to write over the other's artifacts (ml/ml.md#the-texture-ab).


def test_default_variant_keys_are_byte_identical_to_the_unversioned_ones() -> None:
    """Adding variants must not move a single existing blob."""
    assert converted_key(UID) == converted_key(UID, DEFAULT_VARIANT)
    assert converted_key(UID) == f"processed/converted/{UID}.ply"
    assert normalized_key(UID) == f"processed/normalized/{UID}.ply"
    assert view_key(UID, 3) == f"processed/renders/{UID}/view_03.png"


def test_textured_variant_uses_its_own_paths_and_keeps_glb() -> None:
    """PLY carries no UVs, so the textured arm stays in GLB through both stages."""
    assert converted_key(UID, TEXTURED_VARIANT) == f"processed/converted_textured/{UID}.glb"
    assert normalized_key(UID, TEXTURED_VARIANT) == f"processed/normalized_textured/{UID}.glb"
    assert view_key(UID, 3, TEXTURED_VARIANT) == f"processed/renders_textured/{UID}/view_03.png"


def test_variant_keys_never_collide_with_the_default_arm() -> None:
    """The control arm's ~12k shape-only renders must survive the treatment arm."""
    for variant_key, default_key in (
        (converted_key(UID, TEXTURED_VARIANT), converted_key(UID)),
        (normalized_key(UID, TEXTURED_VARIANT), normalized_key(UID)),
        (view_key(UID, 0, TEXTURED_VARIANT), view_key(UID, 0)),
    ):
        assert variant_key != default_key


@pytest.mark.parametrize("prefix", [CONVERTED_PREFIX, NORMALIZED_PREFIX, RENDERS_PREFIX])
def test_textured_keys_fall_outside_the_reconcile_families(prefix: str) -> None:
    """`processed/converted/` must not prefix-match `processed/converted_textured/`.

    This is what keeps `app.reconcile_from_storage` from rebuilding `artifact` rows
    out of textured blobs. The `artifact` table is unique on `(model_uid, stage)`,
    so such a row would overwrite the one the control arm's trainability query
    joins against — the failure the parallel namespace exists to prevent.
    """
    for key in (
        converted_key(UID, TEXTURED_VARIANT),
        normalized_key(UID, TEXTURED_VARIANT),
        view_key(UID, 0, TEXTURED_VARIANT),
    ):
        assert not key.startswith(prefix)


def test_uid_from_key_is_variant_blind() -> None:
    """Deliberate: a variant blob is not a pipeline artifact and owns no DB row."""
    assert uid_from_key(converted_key(UID, TEXTURED_VARIANT)) is None
    assert uid_from_key(view_key(UID, 0, TEXTURED_VARIANT)) is None


def test_view_keys_follow_their_variant() -> None:
    keys = view_keys(UID, TEXTURED_VARIANT)
    assert len(keys) == NUM_VIEWS
    assert all(key.startswith(renders_prefix(UID, TEXTURED_VARIANT)) for key in keys)


@pytest.mark.parametrize("build_key", [converted_key, normalized_key, renders_prefix])
def test_an_unknown_variant_is_refused_not_defaulted(build_key) -> None:
    """A typo that silently fell back to the default paths would have one arm
    writing over the other's artifacts — the one failure that must be loud."""
    with pytest.raises(ValueError, match="unknown artifact variant"):
        build_key(UID, "textured_v2")
