"""FR-3 pass 2 — Sketchfab weak labeling over the full corpus.

Assigns a roster class to each object from its raw Sketchfab metadata. Two
stages (see `ml/ml.md#class-list-approach`):

1. **Category gate** — `taxonomy.SKETCHFAB_CATEGORY_TO_CLASSES` maps the coarse
   top-level `categories` field to the candidate roster classes under it. One
   candidate -> label directly; none -> unlabeled; several -> ambiguous.
2. **Keyword resolution** — `taxonomy.CLASS_TO_KEYWORDS` tag/title keywords pick a
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
import re
from collections import Counter
from pathlib import Path

import objaverse
from io_utils import write_json
from taxonomy import (
    CLASS_TO_KEYWORDS,
    CONFIRM_REQUIRED_CLASSES,
    SKETCHFAB_CATEGORY_TO_CLASSES,
)

_TOKEN = re.compile(r"[a-z0-9]+")


def sample_uids_by_shard(shard_count: int) -> tuple[list[str], list[str]]:
    """Return (uids, shard_ids) for the first `shard_count` whole metadata shards."""
    uid_to_path: dict[str, str] = objaverse._load_object_paths()
    shard_id_to_uids: dict[str, list[str]] = {}
    for uid, path in uid_to_path.items():
        shard_id_to_uids.setdefault(path.split("/")[1], []).append(uid)
    shard_ids = sorted(shard_id_to_uids)[:shard_count]
    return [uid for shard_id in shard_ids for uid in shard_id_to_uids[shard_id]], shard_ids


def category_candidates(annotation: dict) -> set[str]:
    """Roster classes implied by an object's Sketchfab `categories` (the gate)."""
    candidates_set: set[str] = set()
    for category in annotation.get("categories") or []:
        candidates_set.update(SKETCHFAB_CATEGORY_TO_CLASSES.get(category.get("name"), []))
    return candidates_set


def _tokens(annotation: dict) -> set[str]:
    """Lowercase word tokens from the object's title + tag names."""
    text_fields = [annotation.get("name") or ""]
    text_fields += [tag.get("name") or "" for tag in annotation.get("tags") or []]
    return {token for field in text_fields for token in _TOKEN.findall(field.lower())}


def resolve_by_keywords(annotation: dict, candidates_set: set[str]) -> str | None:
    """Pick one class from `candidates_set` by keyword hits; None if no clear winner.

    Scores each candidate by how many of its `CLASS_TO_KEYWORDS` appear in the
    object's tokens; returns the sole top scorer, or None on a zero or tied top
    (stay conservative — an unresolved object is better than a wrong label).
    """
    tokens_set = _tokens(annotation)
    class_to_score = {
        class_name: sum(keyword in tokens_set for keyword in CLASS_TO_KEYWORDS.get(class_name, []))
        for class_name in candidates_set
    }
    top_score = max(class_to_score.values())
    if top_score == 0:
        return None
    winners = [class_name for class_name, score in class_to_score.items() if score == top_score]
    return winners[0] if len(winners) == 1 else None


def label_object(annotation: dict) -> tuple[str | None, str]:
    """Weak-label one object. Returns (class_or_None, reason).

    Single-candidate category -> its class directly, unless that class is
    confirm-required (its category is a noisy grab-bag) in which case a keyword
    must confirm it. Multi-candidate -> keyword resolution (reason "keyword"), or
    "ambiguous" if keywords pick no clear winner; no mapped category ->
    "out-of-scope".
    """
    candidates_set = category_candidates(annotation)
    if not candidates_set:
        return None, "out-of-scope"
    if len(candidates_set) == 1:
        only_class = next(iter(candidates_set))
        if only_class not in CONFIRM_REQUIRED_CLASSES:
            return only_class, "category"
        # confirm-required: fall through to require a keyword match
    resolved_class = resolve_by_keywords(annotation, candidates_set)
    if resolved_class is not None:
        return resolved_class, "keyword"
    return None, "ambiguous"


def run(out_dir: Path, shard_count: int) -> dict[str, object]:
    sample_uids, shard_ids = sample_uids_by_shard(shard_count)
    print("=== Sketchfab weak labels — category gate + keyword resolution ===")
    print(f"shards {', '.join(shard_ids)} ({len(sample_uids):,} objects); "
          f"downloading metadata ...")
    uid_to_annotation = objaverse.load_annotations(sample_uids)

    reason_to_count: Counter[str] = Counter()
    class_to_count: Counter[str] = Counter()
    for annotation in uid_to_annotation.values():
        class_name, reason = label_object(annotation)
        reason_to_count[reason] += 1
        if class_name is not None:
            class_to_count[class_name] += 1

    object_count = len(uid_to_annotation)
    print(f"\nof {object_count:,} objects:")
    for reason in ("category", "keyword", "ambiguous", "out-of-scope"):
        count = reason_to_count[reason]
        print(f"  {count:6,}  ({count / object_count * 100:4.0f}%)  {reason}")
    print("\nlabeled (category + keyword), per class:")
    for class_name, count in class_to_count.most_common():
        print(f"  {count:6,}  {class_name}")

    result: dict[str, object] = {
        "shard_ids": shard_ids,
        "sample_size": len(sample_uids),
        "n_annotations": object_count,
        "by_reason": dict(reason_to_count),
        "labeled_by_category": dict(class_to_count),
    }
    out_path = out_dir / "weak_label_coverage.json"
    write_json(out_path, result)
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
