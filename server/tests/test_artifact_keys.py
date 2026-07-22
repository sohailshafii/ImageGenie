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
