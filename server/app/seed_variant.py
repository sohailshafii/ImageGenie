"""Republish already-ingested models into a variant arm (post-v1 backlog item 1).

The texture A/B needs a second set of artifacts for models the pipeline has
*already* processed: same meshes, different convert/render behaviour, parallel
keys (`app/artifact_keys.py`). This publishes the jobs that produce them.

**It starts at convert, not download.** These uids were ingested long ago and
their raw GLB is already in the bucket, so re-downloading would spend the model
store's bandwidth and our own to fetch bytes we hold. Convert reads
``raw/<uid>.glb`` and writes ``processed/converted_textured/<uid>.glb``, and each
stage hands the variant to the next (`queue.publish_next`), so one convert job per
model drives the whole arm.

    python -m app.seed_variant --from-csv data/experiments/textured_subset.csv --limit 8
    python -m app.seed_variant --from-csv data/experiments/textured_subset.csv \\
                               --from-csv data/devset/lvis_textured.csv

`--limit` takes the file's **first** N rows rather than a sample. The selections
are written in sorted uid order, so a prefix is stable: a pilot of 8 is the first 8
of the eventual full run, and re-running with a larger limit re-publishes those 8
rather than choosing 8 different models. Re-publishing costs nothing — every stage
skips a model whose variant blob already exists (NFR-2).

**Cost lives here.** One job is one model through three stages, ~12 renders at the
end; the full A/B is 3,090 + 587 models. Pilot first and *look at the PNGs* — the
unit tests mock the GL call, so only a real render proves the texture survived.

Reads a plain CSV with a ``uid`` column, which is what both selections are
(`ml/build_texture_subset.py`). Deliberately not an import of that module: the API
image ships without the ml package, and `app` importing `ml` is the layering that
broke the first Vertex evaluation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .artifact_keys import DEFAULT_VARIANT, TEXTURED_VARIANT, VARIANT_TO_LAYOUT
from .config import get_settings
from .queue import publish_next


def read_uids(csv_paths: list[Path], limit: int | None = None) -> list[str]:
    """The uid column of each CSV, in file order, de-duplicated, capped at `limit`.

    De-duplicated because the two selections are read together and a uid could in
    principle appear in both — publishing it twice would not break anything (the
    stages are idempotent) but it would make the printed count a lie.
    """
    uids: list[str] = []
    seen_uids_set: set[str] = set()
    for path in csv_paths:
        if not path.exists():
            raise SystemExit(f"{path} not found — nothing to publish")
        with path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None or "uid" not in reader.fieldnames:
                raise SystemExit(f"{path} has no `uid` column; found {reader.fieldnames}")
            for row in reader:
                if row["uid"] not in seen_uids_set:
                    seen_uids_set.add(row["uid"])
                    uids.append(row["uid"])
    return uids[:limit] if limit is not None else uids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-csv", type=Path, action="append", required=True,
                        dest="csv_paths", metavar="PATH",
                        help="a CSV with a `uid` column; repeatable")
    parser.add_argument("--variant", default=TEXTURED_VARIANT,
                        choices=sorted(VARIANT_TO_LAYOUT),
                        help=f"which arm to build (default: {TEXTURED_VARIANT})")
    parser.add_argument("--limit", type=int, default=None,
                        help="publish only the first N uids — pilot before a full run")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be published and exit")
    args = parser.parse_args()

    if args.variant == DEFAULT_VARIANT:
        # Not a typo-guard but a data-loss guard: the default arm writes over
        # `processed/converted/` and the `artifact` rows the whole catalog reads,
        # which is a reconciliation job, not a seeding one.
        raise SystemExit(
            "refusing to republish the default arm — that rewrites the artifacts "
            "every existing query depends on. Name a variant."
        )

    uids = read_uids(args.csv_paths, args.limit)
    settings = get_settings()
    sources = ", ".join(str(path) for path in args.csv_paths)
    if args.dry_run:
        print(f"would publish {len(uids):,} {args.variant} convert jobs from {sources}")
        print("first: " + ", ".join(uids[:5]))
        return

    for uid in uids:
        publish_next(settings.convert_topic, uid, args.variant)
    print(
        f"published {len(uids):,} {args.variant} convert jobs to "
        f"'{settings.convert_topic}' from {sources}"
    )


if __name__ == "__main__":
    main()
