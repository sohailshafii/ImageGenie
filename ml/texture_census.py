"""Census what colour the raw meshes actually carry (post-v1 backlog item 1, step 0).

The texture A/B asks whether colour and material, rather than shape alone, move macro
recall. Before any of that is built there is a cheaper question: **how many of our
models carry colour at all?** If the answer is "few", the treatment arm is mostly
identical to the control and the experiment measures nothing — while still producing a
plausible-looking number. This module answers it, and in doing so also *sizes* the
experiment: the qualifying uids it reports are the pool the subset is drawn from.

**It does not download the meshes.** A GLB is a 12-byte header, an 8-byte chunk header,
a JSON chunk, and then the binary chunk holding geometry and texture pixels. Everything
that says whether a model has colour — ``images``, ``textures``, each material's
``baseColorTexture`` and ``baseColorFactor``, and each primitive's ``TEXCOORD_0`` /
``COLOR_0`` attributes — lives in that JSON chunk at the *head* of the file. A ranged
read of the header plus the JSON chunk is roughly 1-2% of the bytes (measured: 21.9 KB
of a 1.1 MB object, 1.3 MB of a 28.7 MB one), which is what makes censusing the whole
corpus cheaper than sampling a couple of hundred meshes.

The tier a model lands in is a *reporting label* over the raw counts, not the
measurement itself. Every count is written out per model, so a revised qualifying rule
(which tiers count as "carries colour") is re-derived from the CSV rather than by
re-reading the bucket.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from build_dev_set import DEV_SET_PATH
from io_utils import write_csv, write_json
from sqlalchemy import select

from app.artifact_keys import raw_key
from app.config import get_settings
from app.db import session_scope
from app.models import Artifact, ArtifactStage, ArtifactStatus, Label, Model
from app.storage import Storage, build_storage

# A GLB starts with magic + version + total length (12 bytes), then the first chunk's
# length + type (8 bytes). Reading these 20 bytes is enough to say how long the JSON
# chunk is, and therefore how many more bytes to ask the bucket for.
GLB_MAGIC = b"glTF"
JSON_CHUNK_TYPE = 0x4E4F534A  # b"JSON" little-endian, per the glTF 2.0 spec
HEADER_LENGTH = 20

# glTF's default base colour is opaque white: a material that omits `baseColorFactor`
# carries no colour information, and neither does one that spells the default out.
DEFAULT_BASE_COLOR = (1.0, 1.0, 1.0)
# Base colours are compared as rounded triples so float noise in an exporter's output
# ("0.9999999" vs "1.0") cannot invent a second distinct colour.
COLOR_PRECISION = 4

# Tiers, strongest first. A model is assigned the first one it reaches.
TIER_TEXTURE = "texture"
TIER_VERTEX_COLOUR = "vertex_colour"
TIER_MULTI_COLOUR = "multi_colour"
TIER_UNIFORM_COLOUR = "uniform_colour"
TIER_NONE = "none"
# Not produced here: the reader assigns it when the bytes cannot be parsed at all, so
# an unreadable object is never silently counted as "no colour".
TIER_UNREADABLE = "unreadable"


@dataclass(frozen=True)
class ColourProfile:
    """What one model carries, as counts plus the tier derived from them.

    The counts are the measurement and the tier is a view of them, which is why both
    are stored. `distinct_base_color_count` counts *distinct* rounded RGB triples
    across the materials, so six materials sharing one colour read as one.
    """

    tier: str
    image_count: int
    texture_count: int
    material_count: int
    base_color_texture_count: int
    distinct_base_color_count: int
    has_non_default_base_color: bool
    has_texcoord: bool
    has_vertex_colour: bool


def json_chunk_length(header: bytes) -> int:
    """Bytes of JSON chunk that follow `header`, the first `HEADER_LENGTH` of a GLB.

    Raises ``ValueError`` if the payload is not a GLB whose first chunk is JSON. That
    is a refusal rather than a guess on purpose: the whole census rests on this parse,
    and a file we cannot read must be reported as unreadable, never as uncoloured.
    """
    if len(header) < HEADER_LENGTH:
        raise ValueError(f"need {HEADER_LENGTH} header bytes, got {len(header)}")
    magic, _version, _total_length = struct.unpack("<4sII", header[:12])
    if magic != GLB_MAGIC:
        raise ValueError(f"not a GLB (magic {magic!r})")
    chunk_length, chunk_type = struct.unpack("<II", header[12:HEADER_LENGTH])
    if chunk_type != JSON_CHUNK_TYPE:
        raise ValueError(f"first chunk is not JSON (type {chunk_type:#x})")
    return chunk_length


def parse_gltf_json(header_and_json: bytes) -> dict:
    """The glTF document from a GLB's leading bytes (header + the whole JSON chunk).

    `header_and_json` must be at least ``HEADER_LENGTH + json_chunk_length(...)`` long;
    a short read raises ``ValueError`` rather than parsing a truncated document, which
    would otherwise surface as a model that mysteriously has no materials.
    """
    chunk_length = json_chunk_length(header_and_json)
    chunk_end = HEADER_LENGTH + chunk_length
    if len(header_and_json) < chunk_end:
        raise ValueError(
            f"JSON chunk is {chunk_length} bytes but only "
            f"{len(header_and_json) - HEADER_LENGTH} were read"
        )
    return json.loads(header_and_json[HEADER_LENGTH:chunk_end])


def _base_color(material: dict) -> tuple[float, float, float]:
    """One material's base colour as a rounded RGB triple; the glTF default if absent.

    Alpha is dropped: a transparent white is still white, and the census is about what
    the renderer's grey override throws away, which is hue rather than opacity.
    """
    factor = material.get("pbrMetallicRoughness", {}).get("baseColorFactor")
    if not factor:
        return DEFAULT_BASE_COLOR
    return tuple(round(float(channel), COLOR_PRECISION) for channel in factor[:3])


def colour_profile(gltf: dict) -> ColourProfile:
    """Count `gltf`'s colour channels and assign the strongest tier it reaches.

    The tiers are ordered by how much the CNN could plausibly get from them: a texture
    image beats per-vertex colour, which beats several flat material colours, which
    beats one flat colour, which beats an all-white model that the render stage's grey
    override costs nothing to apply to.
    """
    materials = gltf.get("materials", [])
    primitives = [
        primitive
        for mesh in gltf.get("meshes", [])
        for primitive in mesh.get("primitives", [])
    ]
    attribute_names_set = {
        name for primitive in primitives for name in primitive.get("attributes", {})
    }

    base_color_texture_count = sum(
        1
        for material in materials
        if "baseColorTexture" in material.get("pbrMetallicRoughness", {})
    )
    base_colors_set = {_base_color(material) for material in materials}
    has_texcoord = "TEXCOORD_0" in attribute_names_set
    has_vertex_colour = "COLOR_0" in attribute_names_set
    has_non_default_base_color = any(
        base_color != DEFAULT_BASE_COLOR for base_color in base_colors_set
    )

    # A `baseColorTexture` with no UVs to sample it through is a texture the renderer
    # cannot use, so it does not earn the top tier — it falls through to whatever else
    # the model carries.
    if base_color_texture_count and has_texcoord:
        tier = TIER_TEXTURE
    elif has_vertex_colour:
        tier = TIER_VERTEX_COLOUR
    elif len(base_colors_set) > 1:
        tier = TIER_MULTI_COLOUR
    elif has_non_default_base_color:
        tier = TIER_UNIFORM_COLOUR
    else:
        tier = TIER_NONE

    return ColourProfile(
        tier=tier,
        image_count=len(gltf.get("images", [])),
        texture_count=len(gltf.get("textures", [])),
        material_count=len(materials),
        base_color_texture_count=base_color_texture_count,
        distinct_base_color_count=len(base_colors_set),
        has_non_default_base_color=has_non_default_base_color,
        has_texcoord=has_texcoord,
        has_vertex_colour=has_vertex_colour,
    )


# One ranged read big enough for almost every JSON chunk (measured: 14-22 KB is
# typical, though a 1,293-primitive scene ran to 1.3 MB). Reading this much up front
# means one request per object in the common case and two in the tail, rather than
# always paying two round-trips to learn the chunk length first.
PROBE_LENGTH = 64 * 1024


@dataclass(frozen=True)
class CensusTarget:
    """One model to census.

    `class_name` is the model's current label for the trainable population and its
    *gold* class for the dev set, which carries no label by design; `population` says
    which, because the gate reads coverage for the two separately.
    """

    uid: str
    class_name: str
    raw_key: str
    population: str


@dataclass(frozen=True)
class CensusRow:
    """A target's colour profile, or the reason it could not be read.

    `error` is empty for every readable model. When it is set the tier is
    ``unreadable`` and the counts are zero — an object we could not parse is never
    silently counted as "carries no colour", because those two conclusions would size
    the experiment very differently.
    """

    target: CensusTarget
    profile: ColourProfile
    error: str


UNREADABLE_PROFILE = ColourProfile(
    tier=TIER_UNREADABLE,
    image_count=0,
    texture_count=0,
    material_count=0,
    base_color_texture_count=0,
    distinct_base_color_count=0,
    has_non_default_base_color=False,
    has_texcoord=False,
    has_vertex_colour=False,
)


def read_gltf_json(storage: Storage, key: str) -> dict:
    """The glTF document at the head of the GLB stored at `key`.

    Reads `PROBE_LENGTH` bytes, and only goes back for more when the JSON chunk turns
    out to be longer than that. Raises ``ValueError`` if the object is not a GLB.
    """
    probe = storage.get_range(key, 0, PROBE_LENGTH)
    chunk_end = HEADER_LENGTH + json_chunk_length(probe)
    if len(probe) < chunk_end:
        probe = storage.get_range(key, 0, chunk_end)
    return parse_gltf_json(probe)


def census_one(storage: Storage, target: CensusTarget) -> CensusRow:
    """Census one model, turning any failure into an ``unreadable`` row.

    The exception handler is deliberately broad. This runs unattended over ~13k
    objects, and the failures are a mixed bag — a truncated header, a missing object,
    a transient GCS error — none of which should end the census. Recording the
    exception's text per row keeps them distinguishable in the CSV afterwards, which
    a narrower `except` that let one class of failure through would not.
    """
    try:
        return CensusRow(target, colour_profile(read_gltf_json(storage, target.raw_key)), "")
    except Exception as error:  # noqa: BLE001 — recorded per row, see the docstring
        return CensusRow(target, UNREADABLE_PROFILE, f"{type(error).__name__}: {error}")


def census(
    storage: Storage, targets: Sequence[CensusTarget], num_workers: int, progress_every: int
) -> list[CensusRow]:
    """Census every target concurrently, in input order.

    The work is latency-bound — one or two small ranged GETs per object — so threads
    are the right shape and the pool can be wide. Results come back ordered by target
    rather than by completion, so a re-run over the same input writes the same CSV.
    """
    rows: list[CensusRow] = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for index, row in enumerate(pool.map(partial(census_one, storage), targets), start=1):
            rows.append(row)
            if progress_every and index % progress_every == 0:
                print(f"  censused {index}/{len(targets)}")
    return rows


# The two populations the gate asks about separately: the models a training run can
# actually use, and the independently annotated dev set the headline is scored on.
POPULATION_TRAINABLE = "trainable"
POPULATION_LVIS = "lvis"

CENSUS_DIR = Path("data/census")
CENSUS_PATH = CENSUS_DIR / "texture_census.csv"
SUMMARY_PATH = CENSUS_DIR / "texture_census_summary.json"

CENSUS_HEADER = (
    "uid",
    "population",
    "class",
    "tier",
    "image_count",
    "texture_count",
    "material_count",
    "base_color_texture_count",
    "distinct_base_color_count",
    "has_non_default_base_color",
    "has_texcoord",
    "has_vertex_colour",
    "error",
)

# Tiers in report order, so every table reads the same way.
TIER_ORDER = (
    TIER_TEXTURE,
    TIER_VERTEX_COLOUR,
    TIER_MULTI_COLOUR,
    TIER_UNIFORM_COLOUR,
    TIER_NONE,
    TIER_UNREADABLE,
)


def census_rank(uid: str) -> bytes:
    """A stable sort key for `--limit`, so a pilot censuses the *same* models twice.

    Salted differently from `splits.bucket_of` and `train.subsample_rank` for the
    reason spelled out there: two selection steps sharing a hash select in lockstep,
    and a pilot that happened to draw exactly the test bucket would misreport
    coverage for the models the experiment cares most about.
    """
    return hashlib.sha256(f"census:{uid}".encode()).digest()


def _trainable_targets(session) -> list[CensusTarget]:
    """Live, labeled, rendered models — the same filter `ml/train.py` trains on.

    Deliberately the same three conditions rather than "everything in the bucket":
    the A/B needs both arms, and the control arm reads the *existing* shape-only
    renders, so a model that is unlabeled or unrendered cannot be in either arm and
    censusing it would inflate the qualifying pool with models we cannot use.
    """
    statement = (
        select(Label.model_uid, Label.class_name, Model.raw_key)
        .join(Model, Model.uid == Label.model_uid)
        .join(Artifact, Artifact.model_uid == Label.model_uid)
        .where(Model.deleted_at.is_(None))
        .where(Artifact.stage == ArtifactStage.rendered)
        .where(Artifact.status == ArtifactStatus.done)
        .distinct(Label.model_uid)
        .order_by(Label.model_uid, Label.created_at.desc(), Label.id.desc())
    )
    return [
        CensusTarget(uid, class_name, stored_key or raw_key(uid), POPULATION_TRAINABLE)
        for uid, class_name, stored_key in session.execute(statement).all()
    ]


def _dev_set_targets(session, uid_to_gold_class: dict[str, str]) -> list[CensusTarget]:
    """Rendered dev-set models, carrying their *gold* class rather than a label.

    These uids deliberately have no `label` row — that is what keeps the second dev
    set untrainable — so their class comes from the selection CSV instead
    (ml/build_dev_set.py).
    """
    statement = (
        select(Model.uid, Model.raw_key)
        .join(Artifact, Artifact.model_uid == Model.uid)
        .where(Model.uid.in_(uid_to_gold_class))
        .where(Model.deleted_at.is_(None))
        .where(Artifact.stage == ArtifactStage.rendered)
        .where(Artifact.status == ArtifactStatus.done)
    )
    return [
        CensusTarget(uid, uid_to_gold_class[uid], stored_key or raw_key(uid), POPULATION_LVIS)
        for uid, stored_key in session.execute(statement).all()
    ]


def load_gold_classes(path: Path) -> dict[str, str]:
    """The dev set's ``uid -> gold class`` map, or empty if it has not been selected."""
    if not path.exists():
        print(f"note: {path} not found — skipping the {POPULATION_LVIS} population")
        return {}
    with path.open(newline="", encoding="utf-8") as csv_file:
        return {row["uid"]: row["class"] for row in csv.DictReader(csv_file)}


def load_targets(dev_set_path: Path) -> list[CensusTarget]:
    """Every model worth censusing, across both populations, GLB sources only.

    Non-GLB raws (an admin's STL or OBJ upload) are dropped rather than censused:
    this parser reads glTF, and reporting a format it cannot speak as `unreadable`
    would put a pipeline fact into a column that is supposed to mean "we could not
    tell what colour this model has".
    """
    with session_scope() as session:
        targets = _trainable_targets(session) + _dev_set_targets(
            session, load_gold_classes(dev_set_path)
        )
    glb_targets = [target for target in targets if target.raw_key.endswith(".glb")]
    skipped = len(targets) - len(glb_targets)
    if skipped:
        print(f"note: skipped {skipped} model(s) whose raw mesh is not a GLB")
    return glb_targets


def summarize(rows: Sequence[CensusRow]) -> dict:
    """Per-population tier counts, overall and per class — what the gate reads.

    Structured population → class → tier rather than flattened, because the decision
    is not "how much colour is there" but "is there colour *where colour should
    help*": thin coverage concentrated in `food`, `plant` or `electronics` stops the
    experiment even if the corpus-wide number looks healthy.
    """
    population_to_summary: dict[str, dict] = {}
    for population in (POPULATION_TRAINABLE, POPULATION_LVIS):
        population_rows = [row for row in rows if row.target.population == population]
        if not population_rows:
            continue
        class_to_tier_counts: dict[str, Counter[str]] = {}
        for row in population_rows:
            class_to_tier_counts.setdefault(row.target.class_name, Counter())[
                row.profile.tier
            ] += 1
        population_to_summary[population] = {
            "total": len(population_rows),
            "tier_counts": dict(
                Counter(row.profile.tier for row in population_rows)
            ),
            "class_to_tier_counts": {
                class_name: dict(tier_counts)
                for class_name, tier_counts in sorted(class_to_tier_counts.items())
            },
        }
    return population_to_summary


def print_summary(population_to_summary: dict) -> None:
    """Print the tier breakdown per class, one table per population."""
    column_width = max(len(tier) for tier in TIER_ORDER) + 2
    for population, summary in population_to_summary.items():
        print(f"\n{population} — {summary['total']} models")
        header = "class".ljust(16) + "".join(tier.rjust(column_width) for tier in TIER_ORDER)
        print(header)
        for class_name, tier_counts in summary["class_to_tier_counts"].items():
            counts = "".join(
                str(tier_counts.get(tier, 0)).rjust(column_width) for tier in TIER_ORDER
            )
            print(class_name.ljust(16) + counts)
        totals = "".join(
            str(summary["tier_counts"].get(tier, 0)).rjust(column_width)
            for tier in TIER_ORDER
        )
        print("ALL".ljust(16) + totals)
        # The pool a qualifying rule can draw from at its most generous — every model
        # carrying colour of any kind. The rule itself is chosen against these numbers
        # (plan: "which tiers qualify"), so this is an upper bound, not the answer.
        any_colour = summary["total"] - summary["tier_counts"].get(
            TIER_NONE, 0
        ) - summary["tier_counts"].get(TIER_UNREADABLE, 0)
        share = any_colour / summary["total"] if summary["total"] else 0.0
        print(f"carries colour of any kind: {any_colour}/{summary['total']} ({share:.1%})")


def census_rows_for_csv(rows: Sequence[CensusRow]) -> list[tuple]:
    """Flatten rows to CSV tuples matching `CENSUS_HEADER`."""
    return [
        (
            row.target.uid,
            row.target.population,
            row.target.class_name,
            row.profile.tier,
            row.profile.image_count,
            row.profile.texture_count,
            row.profile.material_count,
            row.profile.base_color_texture_count,
            row.profile.distinct_base_color_count,
            int(row.profile.has_non_default_base_color),
            int(row.profile.has_texcoord),
            int(row.profile.has_vertex_colour),
            row.error,
        )
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="census only this many models (a pilot; chosen by hash, "
                             "so the same models are picked every time)")
    parser.add_argument("--num-workers", type=int, default=16,
                        help="concurrent ranged reads (default: 16)")
    parser.add_argument("--dev-set", type=Path, default=DEV_SET_PATH,
                        help=f"dev-set selection CSV (default: {DEV_SET_PATH})")
    parser.add_argument("--out-dir", type=Path, default=CENSUS_DIR,
                        help=f"where the CSV and summary land (default: {CENSUS_DIR})")
    parser.add_argument("--progress-every", type=int, default=500,
                        help="print progress every N models (0 to silence)")
    args = parser.parse_args()

    settings = get_settings()
    # Named rather than enforced: reading the *local* backend is how the census is
    # exercised in a local smoke, but a run that meant to read prod and silently
    # censused an empty data/storage would otherwise report a corpus with no colour.
    print(f"storage backend: {settings.storage_backend}")

    targets = load_targets(args.dev_set)
    if args.limit is not None:
        targets = sorted(targets, key=lambda target: census_rank(target.uid))[: args.limit]
    print(f"censusing {len(targets)} model(s) with {args.num_workers} workers")

    rows = census(build_storage(settings), targets, args.num_workers, args.progress_every)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    census_path = args.out_dir / CENSUS_PATH.name
    summary_path = args.out_dir / SUMMARY_PATH.name
    write_csv(census_path, CENSUS_HEADER, census_rows_for_csv(rows))
    population_to_summary = summarize(rows)
    write_json(summary_path, population_to_summary)

    print_summary(population_to_summary)
    unreadable = [row for row in rows if row.error]
    if unreadable:
        print(f"\n{len(unreadable)} unreadable; first few:")
        for row in unreadable[:5]:
            print(f"  {row.target.uid}: {row.error}")
    print(f"\nwrote {census_path} and {summary_path}")


if __name__ == "__main__":
    main()
