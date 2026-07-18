"""FR-3 pass 2, stage 3 — evaluate Sketchfab weak labels against the LVIS gold set.

The LVIS-labeled objects are clean, human-curated labels (pass 1). By comparing
the *noisy* Sketchfab weak labels against them on the objects that have both, we
can measure per-class precision/recall of the weak-labeling rules and tune the
keyword lists from data instead of by eyeball. See `ml/ml.md#sketchfab-weak-labeling`.

This commit builds the **gold-label lookup** only: `uid -> roster class`, by
inverting `taxonomy.CLASS_TO_LVIS_CATEGORIES` over the curated LVIS annotations.
Its per-class counts are a sanity check — they must match `build_class_list.py`'s
per-class object counts (same union, keyed the other way).

Metadata only; no meshes downloaded. Output JSON is gitignored (NFR-6).
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import objaverse
from io_utils import write_json
from taxonomy import CLASS_TO_LVIS_CATEGORIES


def build_uid_to_gold_class() -> dict[str, str]:
    """Map each LVIS gold object UID to its roster class (inverse of the merges).

    Our classes are disjoint at the object level (`build_class_list` reports 0
    multi-class objects), so each gold UID resolves to exactly one class. The
    `setdefault` is a defensive first-wins tie-break in case that ever changes.
    """
    category_to_uids: dict[str, list[str]] = objaverse.load_lvis_annotations()
    uid_to_gold_class: dict[str, str] = {}
    for class_name, categories in CLASS_TO_LVIS_CATEGORIES.items():
        for category in categories:
            for uid in category_to_uids.get(category, []):
                uid_to_gold_class.setdefault(uid, class_name)
    return uid_to_gold_class


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/exploration"),
                        help="output directory for the gold-label summary (gitignored)")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    uid_to_gold_class = build_uid_to_gold_class()
    gold_class_to_count = Counter(uid_to_gold_class.values())

    print(f"=== LVIS gold labels: {len(uid_to_gold_class):,} objects, "
          f"{len(gold_class_to_count)} classes ===")
    for class_name, count in gold_class_to_count.most_common():
        print(f"  {count:6,}  {class_name}")

    write_json(args.out_dir / "gold_labels_summary.json", {
        "n_gold_objects": len(uid_to_gold_class),
        "gold_class_to_count": dict(gold_class_to_count),
    })
    print(f"\nwrote {args.out_dir / 'gold_labels_summary.json'}")


if __name__ == "__main__":
    main()
