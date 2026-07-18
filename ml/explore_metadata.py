"""Milestone 1 — Objaverse metadata exploration.

Pull category/tag distributions from Objaverse to choose the final class list
(10-20 well-populated classes) and to set the weak-label support threshold. See
`ml/ml.md#weak-label-policy` and CLAUDE.md milestone 1.

Two data sources:

* **LVIS annotations** — curated per-object categories (~1.1k categories over ~46k
  objects). The clean signal for picking well-populated classes; one small download.
* **Raw Sketchfab annotations (sampled)** — noisy tags/categories over the full ~800k
  corpus; reflects the weak-label source. Optional and sampled to stay cheap (the
  cost guardrail: exercise on a small sample before scaling).

Outputs CSV distributions + a summary JSON under an output dir (gitignored — derived
data is not redistributed, NFR-6) and prints a summary to stdout. Metadata only: no
3D models are downloaded.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import objaverse
from io_utils import write_csv, write_json


def explore_lvis(out_dir: Path, top_n: int, min_support: int) -> dict[str, object]:
    """Rank curated LVIS categories by object count; the class-list signal."""
    category_to_uids: dict[str, list[str]] = objaverse.load_lvis_annotations()
    category_to_count: Counter[str] = Counter(
        {category: len(uids) for category, uids in category_to_uids.items()}
    )
    ranked = category_to_count.most_common()

    write_csv(out_dir / "lvis_category_counts.csv", ("category", "count"), ranked)

    total_assignments = sum(category_to_count.values())
    unique_objects = len({uid for uids in category_to_uids.values() for uid in uids})
    trainable = [(category, count) for category, count in ranked if count >= min_support]

    print("\n=== LVIS categories (curated) ===")
    print(f"categories: {len(category_to_count):,}   assignments: {total_assignments:,}   "
          f"unique objects: {unique_objects:,}")
    print(f"categories with >= {min_support} objects (trainable bar): {len(trainable):,}")
    print(f"\ntop {top_n} categories:")
    for category, count in ranked[:top_n]:
        print(f"  {count:6,}  {category}")

    return {
        "n_categories": len(category_to_count),
        "n_assignments": total_assignments,
        "n_unique_objects": unique_objects,
        "min_support": min_support,
        "n_trainable_categories": len(trainable),
        "top_categories": ranked[:top_n],
        "trainable_categories": trainable,
    }


def explore_raw(out_dir: Path, shard_count: int, top_n: int) -> dict[str, object]:
    """Aggregate raw Sketchfab tags/categories over whole metadata shards.

    Objaverse metadata is 160 shards (~5k objects each). We sample by *whole
    shard* rather than by scattered uid: a scattered uid sample forces a download
    of every shard those uids happen to touch (i.e. almost all of them), whereas
    N whole shards download exactly N files. Deterministic (first N shards) for
    reproducibility (NFR-4).
    """
    uid_to_path: dict[str, str] = objaverse._load_object_paths()  # uid -> "glbs/000-000/uid.glb"
    shard_id_to_uids: dict[str, list[str]] = {}
    for uid, path in uid_to_path.items():
        shard_id_to_uids.setdefault(path.split("/")[1], []).append(uid)

    shard_ids = sorted(shard_id_to_uids)[:shard_count]
    sample_uids = [uid for shard_id in shard_ids for uid in shard_id_to_uids[shard_id]]
    print(f"\n=== Raw Sketchfab annotations "
          f"({len(shard_ids)} shard(s), {len(sample_uids):,} / {len(uid_to_path):,} objects) ===")
    print(f"downloading metadata shards: {', '.join(shard_ids)} ...")
    uid_to_annotation: dict[str, dict] = objaverse.load_annotations(sample_uids)

    category_to_count: Counter[str] = Counter()
    tag_to_count: Counter[str] = Counter()
    for annotation in uid_to_annotation.values():
        for category in annotation.get("categories") or []:
            name = category.get("name") if isinstance(category, dict) else category
            if name:
                category_to_count[str(name)] += 1
        for tag in annotation.get("tags") or []:
            name = tag.get("name") if isinstance(tag, dict) else tag
            if name:
                tag_to_count[str(name)] += 1

    write_csv(out_dir / "raw_sketchfab_categories.csv", ("category", "count"),
              category_to_count.most_common())
    write_csv(out_dir / "raw_sketchfab_tags.csv", ("tag", "count"), tag_to_count.most_common())

    print(f"\ntop {top_n} raw categories:")
    for category, count in category_to_count.most_common(top_n):
        print(f"  {count:6,}  {category}")
    print(f"\ntop {top_n} raw tags:")
    for tag, count in tag_to_count.most_common(top_n):
        print(f"  {count:6,}  {tag}")

    return {
        "shard_ids": shard_ids,
        "sample_size": len(sample_uids),
        "n_annotations_returned": len(uid_to_annotation),
        "n_categories": len(category_to_count),
        "n_tags": len(tag_to_count),
        "top_categories": category_to_count.most_common(top_n),
        "top_tags": tag_to_count.most_common(top_n),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["lvis", "raw", "both"], default="lvis",
                        help="which distribution(s) to compute (default: lvis)")
    parser.add_argument("--out-dir", type=Path, default=Path("data/exploration"),
                        help="output directory for CSVs + summary (gitignored)")
    parser.add_argument("--top", type=int, default=30,
                        help="how many top entries to print (default: 30)")
    parser.add_argument("--min-support", type=int, default=300,
                        help="min objects for a class to clear the trainable bar (ml.md)")
    parser.add_argument("--shards", type=int, default=1,
                        help="raw-mode: number of whole metadata shards to sample "
                             "(~5k objects each, default: 1)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "objaverse_version": getattr(objaverse, "__version__", "unknown"),
        "params": vars(args) | {"out_dir": str(args.out_dir)},
    }

    if args.mode in ("lvis", "both"):
        summary["lvis"] = explore_lvis(args.out_dir, args.top, args.min_support)
    if args.mode in ("raw", "both"):
        summary["raw"] = explore_raw(args.out_dir, args.shards, args.top)

    summary_path = args.out_dir / "summary.json"
    write_json(summary_path, summary)
    print(f"\nwrote CSVs + summary to {args.out_dir}/")


if __name__ == "__main__":
    main()
