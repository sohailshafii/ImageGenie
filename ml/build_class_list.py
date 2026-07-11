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
    n_lvis_objects = len({u for uids in annotations.values() for u in uids})

    assigned: set[str] = set()
    unknown: dict[str, list[str]] = {}
    per_class: list[dict[str, object]] = []

    for cls, cats in LVIS_MERGES.items():
        bad = [c for c in cats if c not in live_keys]
        if bad:
            unknown[cls] = bad
        uids: set[str] = set()
        for c in cats:
            uids.update(annotations.get(c, []))
        assigned.update(cats)
        per_class.append({
            "class": cls,
            "n_objects": len(uids),
            "n_categories": len(cats),
            "clears_bar": len(uids) >= min_support,
        })

    per_class.sort(key=lambda r: r["n_objects"], reverse=True)

    # Objects claimed by more than one class (union double-counts across classes).
    seen: set[str] = set()
    overlap: set[str] = set()
    for cats in LVIS_MERGES.values():
        cls_uids = {u for c in cats for u in annotations.get(c, [])}
        overlap |= seen & cls_uids
        seen |= cls_uids

    unassigned = sorted(
        ((c, len(annotations[c])) for c in live_keys - assigned),
        key=lambda kv: kv[1], reverse=True,
    )[:top_unassigned]

    if unknown:
        print("!! unknown LVIS categories (fix typos in taxonomy.py):")
        for cls, bad in unknown.items():
            print(f"   {cls}: {bad}")

    print(f"\n=== LVIS-merged class support ("
          f"{len(assigned):,}/{len(live_keys):,} categories, "
          f"{len(seen):,}/{n_lvis_objects:,} objects = "
          f"{len(seen) / n_lvis_objects * 100:.0f}% of LVIS) ===")
    print(f"{'class':<14}{'objects':>9}  {'cats':>4}  bar(>={min_support})")
    for r in per_class:
        mark = "PASS" if r["clears_bar"] else "below"
        print(f"{r['class']:<14}{r['n_objects']:>9,}  {r['n_categories']:>4}  {mark}")

    n_pass = sum(r["clears_bar"] for r in per_class)
    print(f"\n{n_pass}/{len(per_class)} classes clear the LVIS bar; "
          f"{len(overlap):,} objects are claimed by >1 class.")
    print("(LVIS is the ~46k clean subset — volume for the bar comes from the "
          "Sketchfab pass; below-bar here = lean on pass 2, not dropped.)")

    print(f"\ntop {top_unassigned} unassigned LVIS categories (candidate omissions):")
    for cat, n in unassigned:
        print(f"  {n:5,}  {cat}")

    result: dict[str, object] = {
        "min_support": min_support,
        "n_classes": len(per_class),
        "n_classes_clear_bar": n_pass,
        "n_lvis_objects": n_lvis_objects,
        "n_objects_assigned": len(seen),
        "n_objects_multi_class": len(overlap),
        "classes": per_class,
        "unknown_categories": unknown,
        "top_unassigned": unassigned,
    }
    out_path = out_dir / "class_list.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
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
