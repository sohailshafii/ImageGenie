"""FR-3 pass 2 — Sketchfab weak labeling over the full corpus.

Assigns a roster class to each object from its raw Sketchfab metadata. Two
stages (see `ml/ml.md#class-list-approach`):

1. **Category gate** — `taxonomy.SKETCHFAB_CATEGORY_CLASSES` maps the coarse
   top-level `categories` field to the candidate roster classes under it. One
   candidate -> label directly; none -> unlabeled; several -> ambiguous.
2. *(next commit)* **Keyword resolution** — tag/title keywords pick a single
   class within a multi-candidate set (e.g. `cars-vehicles` + tag `jet` ->
   aircraft), and disambiguate homographs by category.

This commit implements stage 1 only and reports coverage (out-of-scope /
labeled-by-category / ambiguous) so the ambiguous slice that stage 2 must
resolve is measured before the keyword rules are written.

Samples by *whole metadata shard* — same discipline as `explore_metadata.py`
(scattered-uid sampling forces downloading nearly every shard). Metadata only;
no meshes downloaded. Output JSON is gitignored (NFR-6).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import objaverse
from taxonomy import SKETCHFAB_CATEGORY_CLASSES


def _sample_uids_by_shard(n_shards: int) -> tuple[list[str], list[str]]:
    """Return (uids, shard_ids) for the first `n_shards` whole metadata shards."""
    object_paths: dict[str, str] = objaverse._load_object_paths()
    by_shard: dict[str, list[str]] = {}
    for uid, path in object_paths.items():
        by_shard.setdefault(path.split("/")[1], []).append(uid)
    shard_ids = sorted(by_shard)[:n_shards]
    return [uid for sid in shard_ids for uid in by_shard[sid]], shard_ids


def category_candidates(ann: dict) -> set[str]:
    """Roster classes implied by an object's Sketchfab `categories` (the gate)."""
    cands: set[str] = set()
    for c in ann.get("categories") or []:
        cands.update(SKETCHFAB_CATEGORY_CLASSES.get(c.get("name"), []))
    return cands


def label_object(ann: dict) -> tuple[str | None, str]:
    """Weak-label one object. Returns (class_or_None, reason).

    Stage 1 only: a single-candidate category yields its class; multi-candidate
    returns None/"ambiguous" for stage 2's keyword rules to resolve.
    """
    cands = category_candidates(ann)
    if not cands:
        return None, "out-of-scope"
    if len(cands) == 1:
        return next(iter(cands)), "category"
    return None, "ambiguous"


def run(out_dir: Path, n_shards: int) -> dict[str, object]:
    sample, shard_ids = _sample_uids_by_shard(n_shards)
    print("=== Sketchfab weak labels — stage 1 (category gate) ===")
    print(f"shards {', '.join(shard_ids)} ({len(sample):,} objects); downloading metadata ...")
    anns = objaverse.load_annotations(sample)

    reasons: Counter[str] = Counter()
    per_class: Counter[str] = Counter()
    for ann in anns.values():
        cls, reason = label_object(ann)
        reasons[reason] += 1
        if cls is not None:
            per_class[cls] += 1

    n = len(anns)
    print(f"\nof {n:,} objects:")
    for reason in ("category", "ambiguous", "out-of-scope"):
        c = reasons[reason]
        print(f"  {c:6,}  ({c / n * 100:4.0f}%)  {reason}")
    print("\nlabeled-by-category, per class:")
    for cls, c in per_class.most_common():
        print(f"  {c:6,}  {cls}")

    result: dict[str, object] = {
        "shard_ids": shard_ids,
        "sample_size": len(sample),
        "n_annotations": n,
        "by_reason": dict(reasons),
        "labeled_by_category": dict(per_class),
    }
    out_path = out_dir / "weak_label_coverage.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nwrote {out_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/exploration"),
                        help="output directory for the coverage summary (gitignored)")
    parser.add_argument("--shards", type=int, default=1,
                        help="number of whole metadata shards to sample (~5k each)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run(args.out_dir, args.shards)


if __name__ == "__main__":
    main()
