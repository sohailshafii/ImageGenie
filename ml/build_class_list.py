"""Milestone 1 — lock the class list from the LVIS merge (pass 1 of FR-3).

Applies `ml/taxonomy.py`'s ``CLASS_TO_LVIS_CATEGORIES`` to the curated LVIS annotations and
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
from pathlib import Path

import objaverse
from io_utils import write_json
from taxonomy import CLASS_TO_LVIS_CATEGORIES


def build(out_dir: Path, min_support: int, top_unassigned: int) -> dict[str, object]:
    category_to_uids: dict[str, list[str]] = objaverse.load_lvis_annotations()
    live_category_set = set(category_to_uids)
    n_lvis_objects = len({uid for uids in category_to_uids.values() for uid in uids})

    assigned_categories_set: set[str] = set()
    class_to_unknown_categories: dict[str, list[str]] = {}
    per_class: list[dict[str, object]] = []

    # loop for class_name mapped to list of lvis categories
    # class_name is what we hardcoded
    for class_name, categories in CLASS_TO_LVIS_CATEGORIES.items():
        # get categories that do not correspond to any lvis items
        # categories here are handpicked, which means they can have typos
        unknown_in_class = [
            category for category in categories if category not in live_category_set
        ]
        if unknown_in_class:
            class_to_unknown_categories[class_name] = unknown_in_class
        class_object_uids_set: set[str] = set()
        # for all lvis categories, builds the corresponding set of uids
        for category in categories:
            class_object_uids_set.update(category_to_uids.get(category, []))
        assigned_categories_set.update(categories)
        per_class.append({
            "class": class_name,
            "n_objects": len(class_object_uids_set),
            "n_categories": len(categories),
            "clears_bar": len(class_object_uids_set) >= min_support,
        })

    per_class.sort(key=lambda class_row: class_row["n_objects"], reverse=True)

    # Objects claimed by more than one class (union double-counts across classes).
    assigned_object_uids_set: set[str] = set()
    multi_class_uids_set: set[str] = set()
    for categories in CLASS_TO_LVIS_CATEGORIES.values():
        class_uids_set = {
            uid for category in categories for uid in category_to_uids.get(category, [])
        }
        multi_class_uids_set |= assigned_object_uids_set & class_uids_set
        assigned_object_uids_set |= class_uids_set

    unassigned = sorted(
        (
            (category, len(category_to_uids[category]))
            for category in live_category_set - assigned_categories_set
        ),
        key=lambda entry: entry[1], reverse=True,
    )[:top_unassigned]

    if class_to_unknown_categories:
        print("!! unknown LVIS categories (fix typos in taxonomy.py):")
        for class_name, unknown_in_class in class_to_unknown_categories.items():
            print(f"   {class_name}: {unknown_in_class}")

    print(f"\n=== LVIS-merged class support ("
          f"{len(assigned_categories_set):,}/{len(live_category_set):,} categories, "
          f"{len(assigned_object_uids_set):,}/{n_lvis_objects:,} objects = "
          f"{len(assigned_object_uids_set) / n_lvis_objects * 100:.0f}% of LVIS) ===")
    print(f"{'class':<14}{'objects':>9}  {'cats':>4}  bar(>={min_support})")
    for class_row in per_class:
        mark = "PASS" if class_row["clears_bar"] else "below"
        print(f"{class_row['class']:<14}{class_row['n_objects']:>9,}  "
              f"{class_row['n_categories']:>4}  {mark}")

    n_pass = sum(class_row["clears_bar"] for class_row in per_class)
    print(f"\n{n_pass}/{len(per_class)} classes clear the LVIS bar; "
          f"{len(multi_class_uids_set):,} objects are claimed by >1 class.")
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
        "n_objects_assigned": len(assigned_object_uids_set),
        "n_objects_multi_class": len(multi_class_uids_set),
        "classes": per_class,
        "unknown_categories": class_to_unknown_categories,
        "top_unassigned": unassigned,
    }
    out_path = out_dir / "class_list.json"
    write_json(out_path, result)
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
