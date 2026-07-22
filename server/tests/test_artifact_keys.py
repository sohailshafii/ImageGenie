"""Tests for the artifact key layout, especially the reverse mapping.

`uid_from_key` is what lets the tables be rebuilt from a bucket listing
(server.md#migrations), so the property that matters most is that it inverts the
builders exactly — a round trip, not a hand-written string.
"""

from __future__ import annotations

import pytest

from app.artifact_keys import (
    NUM_VIEWS,
    converted_key,
    file_type_for_raw_key,
    normalized_key,
    raw_key,
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
