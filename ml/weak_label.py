"""FR-3 pass 2 — Sketchfab weak labeling over the full corpus.

Assigns a roster class to each object from its raw Sketchfab metadata. Two
stages (see `ml/ml.md#class-list-approach`):

1. **Category gate** — `taxonomy.SKETCHFAB_CATEGORY_CLASSES` maps the coarse
   top-level `categories` field to the candidate roster classes under it. One
   candidate -> label directly; none -> unlabeled; several -> ambiguous.
2. **Keyword resolution** — `taxonomy.CLASS_KEYWORDS` tag/title keywords pick a
   single class within a multi-candidate set (e.g. `cars-vehicles` + tag `jet`
   -> aircraft). The category gate has already narrowed candidates, so homographs
   disambiguate for free ("jaguar" under `cars-vehicles` only scores car/aircraft).
   No clear winner -> left "ambiguous", not guessed.

Reports coverage by reason (category / keyword / ambiguous / out-of-scope) and
per-class label counts. Out-of-scope objects stay unlabeled for now; rescuing
them by keyword is a later, separately-measured step.

Samples by *whole metadata shard* — same discipline as `explore_metadata.py`
(scattered-uid sampling forces downloading nearly every shard). Metadata only;
no meshes downloaded. Output JSON is gitignored (NFR-6).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import objaverse
from taxonomy import CLASS_KEYWORDS, SKETCHFAB_CATEGORY_CLASSES

_TOKEN = re.compile(r"[a-z0-9]+")


def _sample_uids_by_shard(shard_count: int) -> tuple[list[str], list[str]]:
    """Return (uids, shard_ids) for the first `shard_count` whole metadata shards."""
    object_paths: dict[str, str] = objaverse._load_object_paths()
    uids_by_shard: dict[str, list[str]] = {}
    for uid, path in object_paths.items():
        uids_by_shard.setdefault(path.split("/")[1], []).append(uid)
    shard_ids = sorted(uids_by_shard)[:shard_count]
    return [uid for shard_id in shard_ids for uid in uids_by_shard[shard_id]], shard_ids


def category_candidates(annotation: dict) -> set[str]:
    """Roster classes implied by an object's Sketchfab `categories` (the gate)."""
    candidates: set[str] = set()
    for category in annotation.get("categories") or []:
        candidates.update(SKETCHFAB_CATEGORY_CLASSES.get(category.get("name"), []))
    return candidates


def _tokens(annotation: dict) -> set[str]:
    """Lowercase word tokens from the object's title + tag names."""
    text_fields = [annotation.get("name") or ""]
    text_fields += [tag.get("name") or "" for tag in annotation.get("tags") or []]
    return {token for field in text_fields for token in _TOKEN.findall(field.lower())}


def resolve_by_keywords(annotation: dict, candidates: set[str]) -> str | None:
    """Pick one class from `candidates` by keyword hits; None if no clear winner.

    Scores each candidate by how many of its `CLASS_KEYWORDS` appear in the
    object's tokens; returns the sole top scorer, or None on a zero or tied top
    (stay conservative — an unresolved object is better than a wrong label).
    """
    tokens = _tokens(annotation)
    scores = {
        class_name: sum(keyword in tokens for keyword in CLASS_KEYWORDS.get(class_name, []))
        for class_name in candidates
    }
    top_score = max(scores.values())
    if top_score == 0:
        return None
    winners = [class_name for class_name, score in scores.items() if score == top_score]
    return winners[0] if len(winners) == 1 else None


def label_object(annotation: dict) -> tuple[str | None, str]:
    """Weak-label one object. Returns (class_or_None, reason).

    Single-candidate category -> its class; multi-candidate -> keyword resolution
    (reason "keyword"), or "ambiguous" if keywords pick no clear winner; no
    mapped category -> "out-of-scope".
    """
    candidates = category_candidates(annotation)
    if not candidates:
        return None, "out-of-scope"
    if len(candidates) == 1:
        return next(iter(candidates)), "category"
    resolved_class = resolve_by_keywords(annotation, candidates)
    if resolved_class is not None:
        return resolved_class, "keyword"
    return None, "ambiguous"


def run(out_dir: Path, shard_count: int) -> dict[str, object]:
    sample_uids, shard_ids = _sample_uids_by_shard(shard_count)
    print("=== Sketchfab weak labels — category gate + keyword resolution ===")
    print(f"shards {', '.join(shard_ids)} ({len(sample_uids):,} objects); "
          f"downloading metadata ...")
    annotations = objaverse.load_annotations(sample_uids)

    counts_by_reason: Counter[str] = Counter()
    counts_by_class: Counter[str] = Counter()
    for annotation in annotations.values():
        class_name, reason = label_object(annotation)
        counts_by_reason[reason] += 1
        if class_name is not None:
            counts_by_class[class_name] += 1

    object_count = len(annotations)
    print(f"\nof {object_count:,} objects:")
    for reason in ("category", "keyword", "ambiguous", "out-of-scope"):
        count = counts_by_reason[reason]
        print(f"  {count:6,}  ({count / object_count * 100:4.0f}%)  {reason}")
    print("\nlabeled (category + keyword), per class:")
    for class_name, count in counts_by_class.most_common():
        print(f"  {count:6,}  {class_name}")

    result: dict[str, object] = {
        "shard_ids": shard_ids,
        "sample_size": len(sample_uids),
        "n_annotations": object_count,
        "by_reason": dict(counts_by_reason),
        "labeled_by_category": dict(counts_by_class),
    }
    out_path = out_dir / "weak_label_coverage.json"
    with out_path.open("w", encoding="utf-8") as out_file:
        json.dump(result, out_file, indent=2)
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
