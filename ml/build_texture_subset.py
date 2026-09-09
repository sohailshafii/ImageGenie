"""Select the texture A/B's training subset (post-v1 backlog item 1).

The census answered *whether* the experiment is worth running; this answers *on
which models*. It draws from the census CSV rather than the database or the
bucket, because the census is the only thing that knows which models actually
carry a UV texture — and re-deriving that would mean re-reading 12,767 GLB
headers to reproduce a list that already exists.

**Only the `texture` tier qualifies** (plan decision 8). The pool is 6,816
trainable models, which is ample, so there is no reason to dilute the treatment
arm with flat single colours: every model in the arm gains real surface detail,
and a positive result therefore means textures rather than "some colour".

**Class-balanced, capped at 300 per class** (decision 9), not proportional. At the
hashed 10% test slice a capped class yields ~30 test models, where a proportional
draw would leave `lamp` and `aircraft` with the four to six the run-17 trap is
made of — per-class recall that is noise. Four classes cannot reach the cap and
contribute everything they have; the shortfall is **not** redistributed, because
topping the total back up from the large classes is precisely the imbalance the
cap exists to remove.

Both arms train on this one list, so the A/B is unaffected by any of it. What the
writeup must say plainly is that the subset is texture-only and class-balanced,
so it sits on different class priors than the full corpus: run 15's 0.3712 is
context, not a baseline.

    make texture-subset                  # 3,090 models to data/experiments/
    make texture-subset CAP=100          # a smaller draw, same members at the front

The `class` column is what the model was labeled at selection time and is written
for reporting only — `ml/train.py --subset` re-resolves each uid's current label
through the live query, so a correction between selection and training reaches the
run (the same live-labels property `build_dev_set` deliberately does *not* have,
for the opposite reason: those labels must never enter the database).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from build_dev_set import (
    TEXTURED_DEV_SET_NAME,
    dev_set_path,
    load_dev_set,
    push_dev_set,
)
from io_utils import write_csv
from taxonomy import ROSTER
from texture_census import (
    CENSUS_PATH,
    POPULATION_LVIS,
    POPULATION_TRAINABLE,
    TIER_TEXTURE,
)

from app.artifact_keys import (
    EXPERIMENT_PREFIX,
    NUM_VIEWS,
    TEXTURED_VARIANT,
    experiment_subset_key,
    renders_prefix,
)
from app.config import get_settings
from app.storage import Storage, build_storage

# Where the selection lands. Gitignored with the rest of `data/` (NFR-6).
SUBSET_DIR = Path("data/experiments")
SUBSET_NAME = "textured_subset"
SUBSET_PATH = SUBSET_DIR / f"{SUBSET_NAME}.csv"

# The qualifying rule and the population, imported rather than spelled out: they
# must mean the same thing here as in the census that produced the CSV, and a
# duplicated string literal is how that quietly stops being true.
QUALIFYING_TIER = TIER_TEXTURE
QUALIFYING_POPULATION = POPULATION_TRAINABLE

DEFAULT_CAP_PER_CLASS = 300


def subset_rank(uid: str) -> bytes:
    """A stable ordering key over uids — the selection's only source of order.

    Selection is by **hash of the uid**, never by index into a sorted list. This
    project has had two comparisons destroyed by index-based selection: the split
    that moved 127 models across the test boundary when one label changed, and the
    `--limit` subsample that scored 4 of run 17's 45 held-out models at 0.0%
    (ml.md#why-the-split-is-hashed-not-shuffled). Hashing means a model's
    membership depends on nothing but its own identity, so censusing more models
    later can displace at most one member per class, at the cap boundary.

    Salted distinctly from `splits.bucket_of`, `train.subsample_rank` and
    `census_rank` for the reason those spell out: two selection steps sharing a
    hash select in lockstep, and a subset that happened to align with the test
    bucket would train on exactly what it is scored against, or on none of it.

    Seed-free, like `build_dev_set.hash_order`. A seed would only buy the ability
    to draw a *different* 3,090 models, which is not something this experiment
    wants: the two arms must sit on one list, and that list is worth being able to
    regenerate from the census alone.
    """
    return hashlib.sha256(f"texture-subset:{uid}".encode()).digest()


def load_qualifying_candidates(census_path: Path = CENSUS_PATH) -> dict[str, list[str]]:
    """Read the census and group the qualifying uids by class.

    Rows outside the roster are dropped, matching `train.load_trainable_samples`
    and `build_dev_set.load_dev_set`: the census reports whatever class the label
    query returned, including the `unlabeled` bucket it uses for models with no
    label at all, and only roster classes can be trained on.
    """
    if not census_path.exists():
        raise SystemExit(
            f"{census_path} not found — run `make census` first; it is gitignored "
            "(NFR-6), so a fresh checkout has no copy."
        )

    class_to_candidates: dict[str, list[str]] = {class_name: [] for class_name in ROSTER}
    with census_path.open(newline="", encoding="utf-8") as census_file:
        for row in csv.DictReader(census_file):
            if row["population"] != QUALIFYING_POPULATION or row["tier"] != QUALIFYING_TIER:
                continue
            if row["class"] in class_to_candidates:
                class_to_candidates[row["class"]].append(row["uid"])

    for candidates in class_to_candidates.values():
        candidates.sort(key=subset_rank)
    return class_to_candidates


def select_subset(
    class_to_candidates: dict[str, list[str]], cap: int = DEFAULT_CAP_PER_CLASS
) -> list[tuple[str, str]]:
    """Take the `cap` lowest-ranked uids of each class, in roster order.

    A class with fewer than `cap` candidates contributes all of them — the arm is
    bounded by what carries textures, and `lamp` has 102 textured models in the
    entire corpus, so no composition raises it. Its per-class recall rests on ~10
    test models in both arms and is directional at best; that belongs in the
    writeup rather than in a redistribution rule that would hide it.
    """
    selected: list[tuple[str, str]] = []
    for class_name in ROSTER:
        for uid in class_to_candidates[class_name][:cap]:
            selected.append((uid, class_name))
    return sorted(selected)


def report(
    class_to_candidates: dict[str, list[str]], selected: list[tuple[str, str]], cap: int
) -> None:
    """Print per-class supply and selection, so a class short of the cap is visible."""
    selected_counts = Counter(class_name for _, class_name in selected)

    print(f"\n{'class':<14}{'textured':>10}{'selected':>10}")
    for class_name in ROSTER:
        available = len(class_to_candidates[class_name])
        short = "  (under cap)" if available < cap else ""
        print(f"{class_name:<14}{available:>10,}{selected_counts[class_name]:>10,}{short}")
    print(
        f"\n{sum(len(uids) for uids in class_to_candidates.values()):,} trainable models "
        f"carry a {QUALIFYING_TIER} tier; {len(selected):,} selected at a cap of {cap:,} "
        f"per class."
    )


def load_qualifying_dev_set_uids(census_path: Path = CENSUS_PATH) -> set[str]:
    """The dev-set uids that carry a UV texture, from the census's `lvis` rows."""
    with census_path.open(newline="", encoding="utf-8") as census_file:
        return {
            row["uid"]
            for row in csv.DictReader(census_file)
            if row["population"] == POPULATION_LVIS and row["tier"] == QUALIFYING_TIER
        }


def select_textured_dev_set(
    dev_set: list[tuple[str, str]], qualifying_uids_set: set[str]
) -> list[tuple[str, str]]:
    """The gold selection restricted to models that can be rendered in colour.

    **The headline number depends on this existing.** 587 of the 984 LVIS models
    are texture-tier, so the treatment arm has no textured renders for the other
    397 — and scoring the control on all 984 while the treatment sees 587 would
    put the two arms on different models, which is the failure that has already
    produced two pairs of opposite-looking numbers in this project
    (ml.md#evaluation). Both arms score `lvis_textured`; neither scores `lvis`.

    Restricting rather than accepting partial coverage is deliberate. The
    coverage floor would mark a 59.7% treatment report as partial and let it
    through, which describes the shortfall honestly and still leaves the
    comparison meaningless.
    """
    return [
        (uid, class_name)
        for uid, class_name in dev_set
        if uid in qualifying_uids_set
    ]


def write_textured_dev_set(
    dev_set: list[tuple[str, str]], path: Path | None = None
) -> Path:
    """Write the restricted selection in the dev-set CSV shape and return its path.

    Same `uid,class,reason` columns `build_dev_set` writes, because
    `load_dev_set` reads this file back through exactly the same parser — the
    restriction changes which rows exist, never what a row is.

    It is deliberately **not** marked in `dev_set_member`. Every one of these uids
    is already reserved under the `lvis` selection, and the labeling UI's guard
    asks "is this model reserved at all", not "to which set" (server/app/api.py),
    so a second row per uid would protect nothing that is not already protected.
    """
    path = path or dev_set_path(TEXTURED_DEV_SET_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(
        path,
        ("uid", "class", "reason"),
        [(uid, class_name, "lvis-gold-textured") for uid, class_name in dev_set],
    )
    return path


def load_subset(name: str = SUBSET_NAME, path: Path | None = None) -> list[str]:
    """Read a named selection back as a list of uids — the local file first, the
    stored copy second.

    Named rather than pathed, unlike `build_dev_set.load_dev_set`: a training run
    is given `--subset textured_subset` and must resolve that identically whether
    it is running from a checkout or from a Vertex job, so the name is the
    identifier and both locations are derived from it.

    The same two-place lookup `build_dev_set.load_dev_set` does, and for the same
    reason: the local file is authoritative where it exists because that is what
    `make texture-subset` writes and what an operator would edit, while the bucket
    is how a Vertex job with no checkout reads the identical list.

    **Uids only.** The CSV's `class` column is what the model was labeled at
    selection time; a training run resolves each uid's *current* label through the
    live query instead, so a correction landing between selection and training
    reaches the run. Returning the classes here would make it far too easy to
    train on a frozen copy of them by accident.
    """
    path = path or SUBSET_DIR / f"{name}.csv"
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        text = _read_stored_subset(name, path)
    return [row["uid"] for row in csv.DictReader(io.StringIO(text))]


def _read_stored_subset(name: str, path: Path) -> str:
    """The bucket copy, or an error naming both places it was not found.

    Two different fixes — select the subset, or push it — so a job that cannot
    find its list has to say which one is missing rather than just failing.
    """
    key = experiment_subset_key(name)
    try:
        return build_storage(get_settings()).get_bytes(key).decode("utf-8")
    except Exception as error:  # noqa: BLE001 - every backend fails differently
        raise SystemExit(
            f"no subset at {path} and none stored at {key} ({error}). Build it with "
            "`make texture-subset`, then `make texture-subset-push` to make it "
            "readable from a cloud job. It is gitignored (NFR-6), so a fresh "
            "checkout has neither."
        ) from error


def rendered_uids(storage: Storage, variant: str = TEXTURED_VARIANT) -> set[str]:
    """Uids whose every view exists under `variant`, from one prefix listing.

    A listing rather than `dataset.has_all_views`: that costs a storage HEAD per
    view, which is ~44,000 requests over this experiment's population, and this
    answers the same question in one paginated pass. It also keeps the check out
    of the ml package's torch import, so it can run anywhere the bucket is
    reachable.

    A model with *some* views is deliberately excluded rather than repaired here.
    A half-rendered model would fault mid-epoch inside a DataLoader worker on a
    paid GPU; if the shortfall is transient the fix is to replay its job, not to
    quietly train around it.
    """
    prefix = renders_prefix("", variant).rstrip("/") + "/"
    uid_to_view_count: Counter[str] = Counter()
    for key in storage.list_keys(prefix):
        remainder = key[len(prefix) :]
        uid, separator, view = remainder.partition("/")
        if uid and separator and view.endswith(".png"):
            uid_to_view_count[uid] += 1
    return {uid for uid, count in uid_to_view_count.items() if count >= NUM_VIEWS}


def verify_against_renders(
    paths: Sequence[Path], storage: Storage, variant: str = TEXTURED_VARIANT
) -> dict[Path, int]:
    """Rewrite each CSV down to the models that actually rendered; report the drops.

    **This is the protocol step, not a cleanup.** Convert refuses a model whose
    packed texture atlas exceeds what a renderer accepts, so a uid selected from
    the census can still fail to reach the treatment arm — and a control arm
    trained on the full list against a treatment arm short of it is two arms on
    different populations, which is the failure this whole experiment is designed
    to avoid (ml.md#evaluation). Both arms train on what survives.

    Rewrites in place, preserving each file's own columns, because the two lists
    are read by name from a fixed path and the bucket copy is refreshed from the
    same file.
    """
    surviving_uids_set = rendered_uids(storage, variant)
    path_to_dropped: dict[Path, int] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            header = tuple(reader.fieldnames or ())
            rows = [row for row in reader]
        kept = [row for row in rows if row["uid"] in surviving_uids_set]
        path_to_dropped[path] = len(rows) - len(kept)
        write_csv(path, header, [tuple(row[column] for column in header) for row in kept])
        print(
            f"{path}: {len(kept):,} of {len(rows):,} rendered "
            f"({path_to_dropped[path]:,} dropped)"
        )
    return path_to_dropped


def push_subset(path: Path = SUBSET_PATH, name: str = SUBSET_NAME) -> str:
    """Copy the selection into the processed bucket and return the key it landed on.

    A Vertex job has no checkout and no `data/` directory, so a subset that exists
    only on a laptop can only ever be trained from that laptop — and both arms of
    this A/B run in the cloud.

    Refuses a non-GCS backend for the reason `build_dev_set.push_dev_set` does:
    `storage_backend` defaults to "local", so a push would copy the file into
    `data/storage/`, print success, and change nothing any cloud job can read.
    `make devset-push` did exactly that once.
    """
    if not path.exists():
        raise SystemExit(
            f"{path} not found — nothing to push; run `make texture-subset` first"
        )
    settings = get_settings()
    if settings.storage_backend != "gcs":
        raise SystemExit(
            f"storage backend is {settings.storage_backend!r}, so this would copy the "
            "subset into the local data/storage directory, where no cloud job can "
            "read it. Re-run with IMAGEGENIE_STORAGE_BACKEND=gcs (`make "
            "texture-subset-push` sets it)."
        )
    storage = build_storage(settings)
    key = experiment_subset_key(name)
    storage.put_bytes(key, path.read_bytes())
    print(f"pushed {path} to {key}")
    _list_experiments(storage)
    return key


def _list_experiments(storage: Storage) -> None:
    """List the experiment prefix back, so "pushed" is an observation not a claim.

    The backend guard above rules out the local-directory case, but it cannot say
    the object arrived — a wrong bucket, or a push that silently no-ops, reads the
    same from here. Listing costs one request and turns the success line into
    something checked against the bucket. Wrapped because a listing failure must
    not lose a push that already happened.
    """
    try:
        stored = sorted(storage.list_sizes(EXPERIMENT_PREFIX))
    except Exception as error:  # noqa: BLE001 - every backend fails differently
        print(f"note: could not list {EXPERIMENT_PREFIX} to confirm ({error})")
        return
    print(f"\n{EXPERIMENT_PREFIX} now holds {len(stored)} object(s):")
    for key, size in stored:
        print(f"  {key}  {size:,} bytes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", type=Path, default=CENSUS_PATH,
                        help=f"census CSV to select from (default: {CENSUS_PATH})")
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP_PER_CLASS,
                        help="maximum models per class (default: 300)")
    parser.add_argument("--out", type=Path, default=SUBSET_PATH,
                        help=f"where to write the uid,class CSV (default: {SUBSET_PATH})")
    # Push separately from select, as `build_dev_set` does, but for a different
    # reason: re-selecting here is harmless (the census does not change under it),
    # while pushing a subset the arms are *already training on* would silently
    # redefine the experiment's population mid-flight. Keeping them separate makes
    # "publish this list" a deliberate act.
    pushing = parser.add_mutually_exclusive_group()
    pushing.add_argument("--push", action="store_true",
                         help="also copy the new selection to the processed bucket")
    pushing.add_argument("--verify-renders", action="store_true",
                         help="cut both selections down to the models that actually "
                              "rendered under the variant, then push. Run this after "
                              "the textured pipeline and before either training run")
    pushing.add_argument("--push-only", action="store_true",
                         help="copy the existing subset and dev-set CSVs to the "
                              "bucket and select nothing")
    args = parser.parse_args()

    if args.push_only:
        push_subset(args.out)
        push_dev_set(dev_set_path(TEXTURED_DEV_SET_NAME), TEXTURED_DEV_SET_NAME)
        return

    if args.verify_renders:
        # Run AFTER the textured pipeline, before either training run. Pushing
        # follows in the same breath: a bucket copy still naming models that never
        # rendered is what a cloud job would read.
        settings = get_settings()
        verify_against_renders(
            [args.out, dev_set_path(TEXTURED_DEV_SET_NAME)], build_storage(settings)
        )
        push_subset(args.out)
        push_dev_set(dev_set_path(TEXTURED_DEV_SET_NAME), TEXTURED_DEV_SET_NAME)
        return

    class_to_candidates = load_qualifying_candidates(args.census)
    selected = select_subset(class_to_candidates, args.cap)
    report(class_to_candidates, selected, args.cap)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, ("uid", "class"), selected)
    print(f"\nwrote {args.out}")

    # The scoring half, written in the same breath as the training half on
    # purpose: the two lists together *are* the experiment's population, and a
    # regenerated subset paired with a stale dev set is a comparison nobody would
    # notice was broken.
    dev_set = select_textured_dev_set(
        load_dev_set(), load_qualifying_dev_set_uids(args.census)
    )
    dev_set_file = write_textured_dev_set(dev_set)
    print(f"wrote {dev_set_file} ({len(dev_set)} of the gold set carry a texture)")

    if args.push:
        push_subset(args.out)
        push_dev_set(dev_set_file, TEXTURED_DEV_SET_NAME)


if __name__ == "__main__":
    main()
