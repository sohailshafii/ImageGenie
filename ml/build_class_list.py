"""Milestone 1 — lock the class list from the LVIS merge (pass 1 of FR-3).

Applies `ml/taxonomy.py`'s ``LVIS_MERGES`` to the curated LVIS annotations and
reports, per class, the count of **unique objects** (union of the merged
categories' UIDs — shared objects are not double-counted). This is the clean
"seed + gold set" signal for the roster; it does NOT measure final weak-label
volume, which comes from the Sketchfab rules over the full corpus (pass 2).

So a per-class count here is measured against ~46k LVIS-labeled objects only.
The >=300 bar (`ml/ml.md`) is ultimately judged on the full-corpus Sketchfab
pass; a class below the bar in LVIS is flagged, not dropped. See
`ml/ml.md#class-list-approach`.

Also surfaces two integrity checks: unknown category strings (typos vs. live
LVIS keys) and the biggest LVIS categories left unassigned (candidate omissions).

Writes ``class_list.json`` to the output dir (gitignored, NFR-6) and prints a
summary. Metadata only; no meshes downloaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import objaverse
from taxonomy import LVIS_MERGES


def build(out_dir: Path, min_support: int, top_unassigned: int) -> dict[str, object]:
    annotations: dict[str, list[str]] = objaverse.load_lvis_annotations()
    live_keys = set(annotations)
    n_lvis_objects = len({uid for uids in annotations.values() for uid in uids})

    assigned_categories: set[str] = set()
    unknown_by_class: dict[str, list[str]] = {}
    per_class: list[dict[str, object]] = []

    for class_name, categories in LVIS_MERGES.items():
        unknown_in_class = [c for c in categories if c not in live_keys]
        if unknown_in_class:
            unknown_by_class[class_name] = unknown_in_class
        class_object_uids: set[str] = set()
        for category in categories:
            class_object_uids.update(annotations.get(category, []))
        assigned_categories.update(categories)
        per_class.append({
            "class": class_name,
            "n_objects": len(class_object_uids),
            "n_categories": len(categories),
            "clears_bar": len(class_object_uids) >= min_support,
        })

    per_class.sort(key=lambda class_row: class_row["n_objects"], reverse=True)

    # Objects claimed by more than one class (union double-counts across classes).
    assigned_object_uids: set[str] = set()
    multi_class_uids: set[str] = set()
    for categories in LVIS_MERGES.values():
        class_uids = {uid for c in categories for uid in annotations.get(c, [])}
        multi_class_uids |= assigned_object_uids & class_uids
        assigned_object_uids |= class_uids

    unassigned = sorted(
        ((category, len(annotations[category])) for category in live_keys - assigned_categories),
        key=lambda entry: entry[1], reverse=True,
    )[:top_unassigned]

    if unknown_by_class:
        print("!! unknown LVIS categories (fix typos in taxonomy.py):")
        for class_name, unknown_in_class in unknown_by_class.items():
            print(f"   {class_name}: {unknown_in_class}")

    print(f"\n=== LVIS-merged class support ("
          f"{len(assigned_categories):,}/{len(live_keys):,} categories, "
          f"{len(assigned_object_uids):,}/{n_lvis_objects:,} objects = "
          f"{len(assigned_object_uids) / n_lvis_objects * 100:.0f}% of LVIS) ===")
    print(f"{'class':<14}{'objects':>9}  {'cats':>4}  bar(>={min_support})")
    for class_row in per_class:
        mark = "PASS" if class_row["clears_bar"] else "below"
        print(f"{class_row['class']:<14}{class_row['n_objects']:>9,}  "
              f"{class_row['n_categories']:>4}  {mark}")

    n_pass = sum(class_row["clears_bar"] for class_row in per_class)
    print(f"\n{n_pass}/{len(per_class)} classes clear the LVIS bar; "
          f"{len(multi_class_uids):,} objects are claimed by >1 class.")
    print("(LVIS is the ~46k clean subset — volume for the bar comes from the "
          "Sketchfab pass; below-bar here = lean on pass 2, not dropped.)")

    print(f"\ntop {top_unassigned} unassigned LVIS categories (candidate omissions):")
    for category, count in unassigned:
        print(f"  {count:5,}  {category}")

    result: dict[str, object] = {
        "min_support": min_support,
        "n_classes": len(per_class),
        "n_classes_clear_bar": n_pass,
        "n_lvis_objects": n_lvis_objects,
        "n_objects_assigned": len(assigned_object_uids),
        "n_objects_multi_class": len(multi_class_uids),
        "classes": per_class,
        "unknown_categories": unknown_by_class,
        "top_unassigned": unassigned,
    }
    out_path = out_dir / "class_list.json"
    with out_path.open("w", encoding="utf-8") as out_file:
        json.dump(result, out_file, indent=2)
    print(f"\nwrote {out_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/exploration"),
                        help="output directory for class_list.json (gitignored)")
    parser.add_argument("--min-support", type=int, default=300,
                        help="LVIS-merged objects for a class to clear the bar (ml.md)")
    parser.add_argument("--top-unassigned", type=int, default=25,
                        help="how many top unassigned LVIS categories to list")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    build(args.out_dir, args.min_support, args.top_unassigned)


if __name__ == "__main__":
    main()
