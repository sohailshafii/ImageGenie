"""The GLB header parse and colour tiers (backlog item 1 step 0, ml.md#the-texture-census).

These are the census's load-bearing functions: every coverage number, and therefore
the decision to run the experiment at all, is a count of what `colour_profile` returns.
A silent wrong answer here — counting a texture that is not there, or reading a
truncated document as a model with no materials — would not look like a failure, so
the parse is tested against hand-built GLBs including the malformed cases.
"""

from __future__ import annotations

import json
import struct

import pytest
from texture_census import (
    GLB_MAGIC,
    HEADER_LENGTH,
    JSON_CHUNK_TYPE,
    PROBE_LENGTH,
    TIER_MULTI_COLOUR,
    TIER_NONE,
    TIER_TEXTURE,
    TIER_UNIFORM_COLOUR,
    TIER_UNREADABLE,
    TIER_VERTEX_COLOUR,
    CensusTarget,
    census,
    census_one,
    colour_profile,
    json_chunk_length,
    parse_gltf_json,
    read_gltf_json,
)


def make_glb(gltf: dict, magic: bytes = GLB_MAGIC, chunk_type: int = JSON_CHUNK_TYPE) -> bytes:
    """A GLB's leading bytes: the 12-byte header, the chunk header, and the JSON."""
    payload = json.dumps(gltf).encode()
    header = struct.pack("<4sII", magic, 2, HEADER_LENGTH + len(payload))
    return header + struct.pack("<II", len(payload), chunk_type) + payload


def make_gltf(
    materials: list[dict] | None = None, attributes: list[str] | None = None
) -> dict:
    """A minimal glTF document with one mesh primitive carrying `attributes`."""
    return {
        "materials": materials or [],
        "meshes": [
            {"primitives": [{"attributes": {name: 0 for name in (attributes or ["POSITION"])}}]}
        ],
    }


class FakeStorage:
    """Ranged reads over in-memory blobs; raises on a missing key, as GCS does."""

    def __init__(self, key_to_blob: dict[str, bytes]) -> None:
        self._key_to_blob = key_to_blob
        self.read_lengths: list[int] = []

    def get_range(self, key: str, start: int, length: int) -> bytes:
        if key not in self._key_to_blob:
            raise FileNotFoundError(key)
        self.read_lengths.append(length)
        return self._key_to_blob[key][start : start + length]


def test_json_chunk_length_reads_the_chunk_header():
    glb = make_glb(make_gltf())
    assert json_chunk_length(glb) == len(glb) - HEADER_LENGTH


def test_json_chunk_length_refuses_a_short_header():
    with pytest.raises(ValueError, match="header bytes"):
        json_chunk_length(b"glTF")


def test_json_chunk_length_refuses_a_non_glb():
    with pytest.raises(ValueError, match="not a GLB"):
        json_chunk_length(make_glb(make_gltf(), magic=b"STL "))


def test_json_chunk_length_refuses_a_non_json_first_chunk():
    with pytest.raises(ValueError, match="not JSON"):
        json_chunk_length(make_glb(make_gltf(), chunk_type=0x004E4942))


def test_parse_gltf_json_refuses_a_truncated_chunk():
    """A short read must raise, not parse — otherwise it reads as "no materials"."""
    glb = make_glb(make_gltf(materials=[{"name": "red"}]))
    with pytest.raises(ValueError, match="only"):
        parse_gltf_json(glb[:-10])


def test_texture_tier_needs_uvs_to_sample_through():
    with_texture = [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}]
    textured = make_gltf(with_texture, attributes=["POSITION", "TEXCOORD_0"])
    assert colour_profile(textured).tier == TIER_TEXTURE

    # Same material, no UVs: the renderer cannot sample it, so it falls through.
    unusable = make_gltf(with_texture, attributes=["POSITION"])
    profile = colour_profile(unusable)
    assert profile.tier == TIER_NONE
    assert profile.base_color_texture_count == 1


def test_vertex_colour_tier():
    gltf = make_gltf(attributes=["POSITION", "COLOR_0"])
    assert colour_profile(gltf).tier == TIER_VERTEX_COLOUR


def test_multi_colour_needs_two_distinct_colours_not_two_materials():
    """Six materials that share one colour are not a coloured model."""
    same_colour = [
        {"name": name, "pbrMetallicRoughness": {"roughnessFactor": 0.6}}
        for name in ("a", "b", "c", "d", "e", "f")
    ]
    profile = colour_profile(make_gltf(same_colour))
    assert profile.material_count == 6
    assert profile.distinct_base_color_count == 1
    assert profile.tier == TIER_NONE

    two_colours = [
        {"pbrMetallicRoughness": {"baseColorFactor": [1.0, 0.0, 0.0, 1.0]}},
        {"pbrMetallicRoughness": {"baseColorFactor": [0.0, 0.0, 1.0, 1.0]}},
    ]
    assert colour_profile(make_gltf(two_colours)).tier == TIER_MULTI_COLOUR


def test_uniform_colour_is_its_own_tier():
    one_colour = [{"pbrMetallicRoughness": {"baseColorFactor": [1.0, 0.0, 0.0, 1.0]}}]
    assert colour_profile(make_gltf(one_colour)).tier == TIER_UNIFORM_COLOUR


def test_default_white_is_not_a_colour_and_alpha_is_ignored():
    """Spelling out the glTF default, transparently, still says nothing about hue."""
    transparent_white = [{"pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 0.4]}}]
    profile = colour_profile(make_gltf(transparent_white))
    assert profile.tier == TIER_NONE
    assert not profile.has_non_default_base_color


def test_read_gltf_json_takes_one_read_when_the_chunk_fits():
    storage = FakeStorage({"raw/small.glb": make_glb(make_gltf())})
    assert read_gltf_json(storage, "raw/small.glb")["materials"] == []
    assert storage.read_lengths == [PROBE_LENGTH]


def test_read_gltf_json_goes_back_for_an_oversized_chunk():
    """The 1.3 MB JSON chunk case: the probe is short, so it re-reads exactly."""
    padded = make_gltf(materials=[{"name": "x" * PROBE_LENGTH}])
    glb = make_glb(padded)
    storage = FakeStorage({"raw/big.glb": glb})
    assert read_gltf_json(storage, "raw/big.glb")["materials"][0]["name"].startswith("x")
    assert storage.read_lengths == [PROBE_LENGTH, len(glb)]


def test_census_one_records_a_failure_instead_of_raising():
    """One unreadable object must not end a 13k-object census, or read as uncoloured."""
    target = CensusTarget("missing", "chair", "raw/missing.glb", "trainable")
    row = census_one(FakeStorage({}), target)
    assert row.profile.tier == TIER_UNREADABLE
    assert "FileNotFoundError" in row.error


def test_census_preserves_input_order():
    """Results follow the targets, not completion order, so re-runs match byte for byte."""
    key_to_blob = {
        f"raw/{index}.glb": make_glb(make_gltf(attributes=["POSITION", "COLOR_0"]))
        for index in range(5)
    }
    targets = [
        CensusTarget(str(index), "chair", f"raw/{index}.glb", "trainable")
        for index in range(5)
    ]
    rows = census(FakeStorage(key_to_blob), targets, num_workers=4, progress_every=0)
    assert [row.target.uid for row in rows] == [str(index) for index in range(5)]
    assert {row.profile.tier for row in rows} == {TIER_VERTEX_COLOUR}
